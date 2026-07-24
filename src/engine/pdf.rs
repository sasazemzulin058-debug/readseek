// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

//! Page-oriented PDF extraction.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::ops::Range;
use std::path::Path;

use anyhow::{Context, Result, bail};
use pdf_oxide::extractors::images::{ImageData, PdfImage, PixelFormat};
use pdf_oxide::{Destination, OutlineItem, PdfDocument, RegionRole};
use serde::Serialize;

use crate::engine::document::{
    Asset, BoundingBox, Document, DocumentFormat, Node, NodeKind, SourceAnchor,
};
use crate::engine::output::ImageMode;
use crate::engine::qwen::VisionInput;
use crate::engine::vision::{Analysis, DetectedObject, Request};

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct PdfInfo {
    pub(crate) pages: usize,
}

impl PdfInfo {
    pub(crate) fn probe(bytes: &[u8]) -> Result<Self> {
        let document = PdfDocument::from_bytes(bytes.to_vec()).context("parse PDF")?;
        let pages = document.page_count().context("read PDF page count")?;
        Ok(PdfInfo { pages })
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct ReadPdfOutput {
    format: &'static str,
    pages: usize,
    markdown: String,
    images: Vec<PdfImageOutput>,
}

#[derive(Debug, Serialize)]
struct PdfImageOutput {
    page: usize,
    width: u32,
    height: u32,
    mime: &'static str,
    mode: ImageMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    caption: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    objects: Option<Vec<DetectedObject>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ocr: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<String>,
}

struct StructuralSectionState {
    node_id: String,
    node_index: usize,
    headings: Vec<(u8, String)>,
}

#[derive(Clone, Copy)]

struct RegionParents<'a> {
    page_id: &'a str,
    base_parent_id: &'a str,
}

pub(crate) fn extract_document(
    path: &Path,
    bytes: &[u8],
    id: String,
    assets_dir: &Path,
) -> Result<Document> {
    let pdf = PdfDocument::from_bytes(bytes.to_vec()).context("parse PDF")?;
    let pages = pdf.page_count().context("read PDF page count")?;
    let mut document = document_outline(path, &pdf, id, pages)?;
    let mut headings = Vec::new();
    let mut sections = HashMap::new();
    for page_index in 0..pages {
        let page = page_index + 1;
        let structured = pdf
            .extract_structured(page_index)
            .with_context(|| format!("extract structure from PDF page {page}"))?;
        append_page_nodes(
            &document.id,
            page,
            &structured,
            &mut headings,
            &mut sections,
            &mut document.nodes,
        );
        append_page_assets(
            &pdf,
            &document.id,
            page_index,
            page,
            assets_dir,
            &mut document.assets,
        )?;
    }
    Ok(document)
}

fn append_page_assets(
    pdf: &PdfDocument,
    document_id: &str,
    page_index: usize,
    page: usize,
    assets_dir: &Path,
    assets: &mut Vec<Asset>,
) -> Result<()> {
    let handles = pdf
        .page_image_handles(page_index)
        .with_context(|| format!("enumerate images on PDF page {page}"))?;
    if handles.is_empty() {
        return Ok(());
    }
    fs::create_dir_all(assets_dir)
        .with_context(|| format!("create PDF asset directory {}", assets_dir.display()))?;

    for handle in handles {
        let image = match handle.decode() {
            Ok(image) => image,
            Err(error) => {
                tracing::debug!(
                    target: "tracing",
                    "skip undecodable PDF image {} on page {page}: {error}",
                    handle.paint_order
                );
                continue;
            }
        };
        let png = match image.to_png_bytes() {
            Ok(png) if !png.is_empty() => png,
            Ok(_) => continue,
            Err(error) => {
                tracing::debug!(
                    target: "tracing",
                    "skip PDF image {} on page {page}: {error}",
                    handle.paint_order
                );
                continue;
            }
        };
        let id = asset_id(document_id, page, handle.paint_order);
        let path = assets_dir.join(format!("{id}.png"));
        crate::engine::repo::write_atomic(&path, &png)
            .with_context(|| format!("write PDF asset {}", path.display()))?;
        assets.push(Asset {
            id,
            mime: "image/png".to_owned(),
            path,
            source_anchor: Some(SourceAnchor {
                page: Some(page),
                destination: None,
                bbox: Some(BoundingBox {
                    x: handle.bbox.x,
                    y: handle.bbox.y,
                    width: handle.bbox.width,
                    height: handle.bbox.height,
                }),
            }),
        });
    }
    Ok(())
}

fn asset_id(document_id: &str, page: usize, paint_order: usize) -> String {
    let digest = blake3::hash(format!("{document_id}:page:{page}:image:{paint_order}").as_bytes());
    format!("a_{}", &digest.to_hex()[..16])
}

pub(crate) fn extract_outline_document(
    path: &std::path::Path,
    bytes: &[u8],
    id: String,
) -> Result<Document> {
    let pdf = PdfDocument::from_bytes(bytes.to_vec()).context("parse PDF")?;
    let pages = pdf.page_count().context("read PDF page count")?;
    document_outline(path, &pdf, id, pages)
}

fn document_outline(
    path: &std::path::Path,
    pdf: &PdfDocument,
    id: String,
    pages: usize,
) -> Result<Document> {
    let outline = pdf
        .get_outline()
        .context("read PDF outline")?
        .unwrap_or_default();
    let mut nodes = Vec::new();
    append_outline_nodes(&id, &outline, None, "", &mut nodes);

    let mut document = Document {
        id,
        format: DocumentFormat::Pdf,
        source: std::path::PathBuf::new(),
        title: String::new(),
        pages,
        nodes,
        assets: Vec::new(),
    };
    document.rebind_source(path);
    Ok(document)
}

fn append_outline_nodes(
    document_id: &str,
    items: &[OutlineItem],
    parent_id: Option<&str>,
    parent_position: &str,
    nodes: &mut Vec<Node>,
) {
    for (index, item) in items.iter().enumerate() {
        let position = if parent_position.is_empty() {
            index.to_string()
        } else {
            format!("{parent_position}.{index}")
        };
        let id = node_id(document_id, &position);
        let source_anchor = item.dest.as_ref().map(|destination| match destination {
            Destination::PageIndex(page) => SourceAnchor {
                page: Some(page + 1),
                destination: None,
                bbox: None,
            },
            Destination::Named(destination) => SourceAnchor {
                page: None,
                destination: Some(destination.clone()),
                bbox: None,
            },
        });
        nodes.push(Node {
            id: id.clone(),
            parent_id: parent_id.map(str::to_owned),
            kind: NodeKind::Section,
            title: Some(item.title.clone()),
            text: None,
            level: None,
            column: None,
            source_anchor,
        });
        append_outline_nodes(document_id, &item.children, Some(&id), &position, nodes);
    }
}

fn append_page_nodes(
    document_id: &str,
    page: usize,
    structured: &pdf_oxide::StructuredPage,
    headings: &mut Vec<(u8, String)>,
    sections: &mut HashMap<usize, StructuralSectionState>,
    nodes: &mut Vec<Node>,
) {
    let page_id = node_id(document_id, &format!("page:{page}"));
    nodes.push(Node {
        id: page_id.clone(),
        parent_id: None,
        kind: NodeKind::Page,
        title: Some(format!("Page {page}")),
        text: None,
        level: None,
        column: None,
        source_anchor: Some(SourceAnchor {
            page: Some(page),
            destination: None,
            bbox: Some(BoundingBox {
                x: 0.0,
                y: 0.0,
                width: structured.page_width,
                height: structured.page_height,
            }),
        }),
    });

    for (position, region) in structured.regions.iter().enumerate() {
        if let Some(section_id) = region.section_id {
            let state = sections.entry(section_id).or_insert_with(|| {
                let node_id = node_id(document_id, &format!("section:{section_id}"));
                let node_index = nodes.len();
                let title = match region.kind {
                    RegionRole::StructuralHeading { .. } => region.text.trim().to_owned(),
                    _ => format!("Section {section_id}"),
                };
                nodes.push(Node {
                    id: node_id.clone(),
                    parent_id: None,
                    kind: NodeKind::StructuralSection,
                    title: Some(title),
                    text: None,
                    level: None,
                    column: None,
                    source_anchor: Some(SourceAnchor {
                        page: Some(page),
                        destination: None,
                        bbox: Some(BoundingBox {
                            x: region.bbox.x,
                            y: region.bbox.y,
                            width: region.bbox.width,
                            height: region.bbox.height,
                        }),
                    }),
                });
                StructuralSectionState {
                    node_id,
                    node_index,
                    headings: Vec::new(),
                }
            });
            if matches!(region.kind, RegionRole::StructuralHeading { .. })
                && nodes[state.node_index]
                    .title
                    .as_deref()
                    .is_some_and(|title| title.starts_with("Section "))
            {
                nodes[state.node_index].title = Some(region.text.trim().to_owned());
            }
            append_region_node(
                document_id,
                page,
                position,
                region,
                RegionParents {
                    page_id: &page_id,
                    base_parent_id: &state.node_id,
                },
                &mut state.headings,
                nodes,
            );
        } else {
            append_region_node(
                document_id,
                page,
                position,
                region,
                RegionParents {
                    page_id: &page_id,
                    base_parent_id: &page_id,
                },
                headings,
                nodes,
            );
        }
    }
}

fn append_region_node(
    document_id: &str,
    page: usize,
    position: usize,
    region: &pdf_oxide::StructuredRegion,
    parents: RegionParents<'_>,
    headings: &mut Vec<(u8, String)>,
    nodes: &mut Vec<Node>,
) {
    let text = region.text.trim();
    if text.is_empty() {
        return;
    }
    let (kind, level) = match region.kind {
        RegionRole::BodyBlock => (NodeKind::Paragraph, None),
        RegionRole::StructuralHeading { level } => (NodeKind::Heading, Some(level)),
        RegionRole::MarginalLabel => (NodeKind::MarginalLabel, None),
        RegionRole::Header => (NodeKind::Header, None),
        RegionRole::Footer => (NodeKind::Footer, None),
        RegionRole::PageNumber => (NodeKind::PageNumber, None),
        RegionRole::Artifact => (NodeKind::Artifact, None),
    };
    let id = node_id(document_id, &format!("page:{page}:region:{position}"));
    let parent_id = if let Some(level) = level {
        while headings
            .last()
            .is_some_and(|(parent_level, _)| *parent_level >= level)
        {
            headings.pop();
        }
        headings
            .last()
            .map_or_else(|| parents.base_parent_id.to_owned(), |(_, id)| id.clone())
    } else if matches!(kind, NodeKind::Paragraph | NodeKind::MarginalLabel) {
        headings
            .last()
            .map_or_else(|| parents.base_parent_id.to_owned(), |(_, id)| id.clone())
    } else {
        parents.page_id.to_owned()
    };
    let (title, text) = if kind == NodeKind::Heading {
        (Some(text.to_owned()), None)
    } else {
        (None, Some(text.to_owned()))
    };
    nodes.push(Node {
        id: id.clone(),
        parent_id: Some(parent_id),
        kind,
        title,
        text,
        level,
        column: region.column_index,
        source_anchor: Some(SourceAnchor {
            page: Some(page),
            destination: None,
            bbox: Some(BoundingBox {
                x: region.bbox.x,
                y: region.bbox.y,
                width: region.bbox.width,
                height: region.bbox.height,
            }),
        }),
    });
    if let Some(level) = level {
        headings.push((level, id));
    }
}

fn node_id(document_id: &str, position: &str) -> String {
    let digest = blake3::hash(format!("{document_id}\0{position}").as_bytes()).to_string();
    format!("n_{}", &digest[..16])
}

fn selected_page_range(page: Option<usize>, pages: usize) -> Result<Range<usize>> {
    let Some(page) = page else {
        return Ok(0..pages);
    };
    if page == 0 {
        bail!("PDF page must be greater than zero");
    }
    if page > pages {
        bail!("PDF page {page} is past the end of the document ({pages} pages)");
    }
    Ok(page - 1..page)
}

fn rgb_data_len(width: u32, height: u32) -> Option<usize> {
    width.checked_mul(height)?.checked_mul(3)?.try_into().ok()
}

fn analyze_pdf_image(
    image: &PdfImage,
    width: u32,
    height: u32,
    page: usize,
    mode: ImageMode,
    request: Request,
    analyze: &mut impl FnMut(VisionInput<'_>, Request) -> Analysis,
) -> Option<(Analysis, Option<crate::engine::image::PreparedImage>)> {
    if let ImageData::Raw {
        pixels,
        format: PixelFormat::RGB,
    } = image.data()
        && rgb_data_len(width, height) != Some(pixels.len())
    {
        tracing::debug!(target: "tracing", "PDF page {page} image skipped: invalid RGB dimensions");
        return None;
    }
    let encode_png = || match image.to_png_bytes() {
        Ok(png) => Some(png),
        Err(error) => {
            tracing::debug!(target: "tracing", "PDF page {page} image skipped: {error}");
            None
        }
    };
    if mode == ImageMode::None {
        let png = encode_png()?;
        let prepared = match crate::engine::image::preprocess(&png) {
            Ok(prepared) => prepared,
            Err(error) => {
                tracing::debug!(target: "tracing", "PDF page {page} image skipped: {error}");
                return None;
            }
        };
        return Some((Analysis::default(), Some(prepared)));
    }
    if let ImageData::Raw {
        pixels,
        format: PixelFormat::RGB,
    } = image.data()
        && image.icc_profile().is_none()
    {
        let input = VisionInput::Rgb {
            width,
            height,
            pixels,
        };
        return Some((analyze(input, request), None));
    }
    let png = encode_png()?;
    Some((analyze(VisionInput::Encoded(&png), request), None))
}

pub(crate) fn read(
    bytes: &[u8],
    mode: ImageMode,
    page: Option<usize>,
    mut analyze: impl FnMut(VisionInput<'_>, Request) -> Analysis,
) -> Result<ReadPdfOutput> {
    let document = PdfDocument::from_bytes(bytes.to_vec()).context("parse PDF")?;
    let pages = document.page_count().context("read PDF page count")?;
    let mut markdown = String::new();
    let mut images = Vec::new();

    let page_range = selected_page_range(page, pages)?;

    for page_index in page_range {
        let page = page_index + 1;
        let text = document
            .extract_text(page_index)
            .with_context(|| format!("extract text from PDF page {page}"))?;
        if !markdown.is_empty() {
            markdown.push('\n');
        }
        writeln!(markdown, "<!-- readseek:page {page} -->").unwrap();
        let text = text.trim();
        if !text.is_empty() {
            markdown.push_str(text);
            markdown.push('\n');
        }

        let handles = document
            .page_image_handles(page_index)
            .with_context(|| format!("enumerate images on PDF page {page}"))?;
        for handle in handles {
            let width = handle.width;
            let height = handle.height;
            let image = match handle.decode() {
                Ok(image) => image,
                Err(error) => {
                    tracing::debug!(target: "tracing", "PDF page {page} image skipped: {error}");
                    continue;
                }
            };
            let request = Request {
                caption: mode.includes_caption(),
                objects: mode.includes_objects(),
                ocr: mode.includes_ocr(),
            };
            let Some((analysis, prepared)) =
                analyze_pdf_image(&image, width, height, page, mode, request, &mut analyze)
            else {
                continue;
            };
            let (mime, encoding, data) = if let Some(prepared) = prepared {
                (prepared.mime, Some("base64"), Some(prepared.data))
            } else {
                ("image/png", None, None)
            };
            let (caption, objects, ocr) = mode.select_analysis(analysis);
            images.push(PdfImageOutput {
                page,
                width,
                height,
                mime,
                mode,
                caption,
                objects,
                ocr,
                encoding,
                data,
            });
        }
    }

    Ok(ReadPdfOutput {
        format: "pdf",
        pages,
        markdown,
        images,
    })
}
