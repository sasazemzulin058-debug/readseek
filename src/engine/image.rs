// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

//! Image detection and metadata. Text/vision analysis lives in
//! [`crate::engine::vision`].

use std::io::Cursor;

use anyhow::{Context, Result};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::{
    DynamicImage, ImageFormat as RasterFormat, codecs::jpeg::JpegEncoder, imageops::FilterType,
};
use serde::Serialize;
use strum_macros::{AsRefStr, Display};

const MAX_LONG_EDGE: u32 = 1568;
const JPEG_QUALITY: u8 = 80;
const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";

/// Return exact diagram labels from draw.io's embedded PNG `mxfile` metadata.
pub(crate) fn embedded_drawio_text(bytes: &[u8]) -> Option<String> {
    let encoded = png_text_chunk(bytes, b"mxfile")?;
    let xml = String::from_utf8(percent_decode(encoded)?).ok()?;
    let mut reader = quick_xml::Reader::from_str(&xml);
    reader.config_mut().trim_text(true);
    let mut labels = Vec::new();

    loop {
        match reader.read_event() {
            Ok(quick_xml::events::Event::Start(cell) | quick_xml::events::Event::Empty(cell))
                if cell.name().as_ref() == b"mxCell" =>
            {
                for attribute in cell.attributes().flatten() {
                    if attribute.key.as_ref() != b"value" {
                        continue;
                    }
                    let value = attribute
                        .decoded_and_normalized_value(
                            quick_xml::XmlVersion::Implicit1_0,
                            reader.decoder(),
                        )
                        .ok()?;
                    let label = plain_drawio_label(&value);
                    if !label.is_empty() {
                        labels.push(label);
                    }
                }
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Ok(_) => {}
            Err(_) => return None,
        }
    }

    (!labels.is_empty()).then(|| labels.join("\n"))
}

fn png_text_chunk<'a>(bytes: &'a [u8], keyword: &[u8]) -> Option<&'a [u8]> {
    if bytes.get(..PNG_SIGNATURE.len())? != PNG_SIGNATURE {
        return None;
    }

    let mut pos = PNG_SIGNATURE.len();
    while pos.checked_add(12)? <= bytes.len() {
        let len = u32::from_be_bytes(bytes.get(pos..pos + 4)?.try_into().ok()?) as usize;
        let data_start = pos.checked_add(8)?;
        let data_end = data_start.checked_add(len)?;
        let chunk_end = data_end.checked_add(4)?;
        if chunk_end > bytes.len() {
            return None;
        }
        if bytes.get(pos + 4..pos + 8)? == b"tEXt" {
            let data = bytes.get(data_start..data_end)?;
            let split = data.iter().position(|byte| *byte == 0)?;
            if data.get(..split)? == keyword {
                return data.get(split + 1..);
            }
        }
        pos = chunk_end;
    }
    None
}

fn percent_decode(value: &[u8]) -> Option<Vec<u8>> {
    let mut decoded = Vec::with_capacity(value.len());
    let mut pos = 0;
    while pos < value.len() {
        if value[pos] != b'%' {
            decoded.push(value[pos]);
            pos += 1;
            continue;
        }
        let high = hex_digit(*value.get(pos + 1)?)?;
        let low = hex_digit(*value.get(pos + 2)?)?;
        decoded.push(high << 4 | low);
        pos += 3;
    }
    Some(decoded)
}

fn hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn plain_drawio_label(value: &str) -> String {
    let value = value
        .replace("<br>", "\n")
        .replace("<br/>", "\n")
        .replace("<br />", "\n")
        .replace("&nbsp;", " ");
    let mut plain = String::with_capacity(value.len());
    let mut in_tag = false;
    for character in value.chars() {
        match character {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => plain.push(character),
            _ => {}
        }
    }
    plain
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// A recognized image format.
#[derive(Clone, Copy, Debug, Serialize, Display, AsRefStr)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub(crate) enum ImageFormat {
    Png,
    Jpeg,
    Gif,
    WebP,
    Bmp,
    Tiff,
    Avif,
    #[allow(dead_code)]
    Heic,
    Ico,
}

impl ImageFormat {
    pub(crate) fn mime_type(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::Gif => "image/gif",
            Self::WebP => "image/webp",
            Self::Bmp => "image/bmp",
            Self::Tiff => "image/tiff",
            Self::Avif => "image/avif",
            Self::Heic => "image/heic",
            Self::Ico => "image/x-icon",
        }
    }
}

impl TryFrom<image::ImageFormat> for ImageFormat {
    type Error = ();

    fn try_from(kind: image::ImageFormat) -> Result<Self, Self::Error> {
        Ok(match kind {
            image::ImageFormat::Png => Self::Png,
            image::ImageFormat::Jpeg => Self::Jpeg,
            image::ImageFormat::Gif => Self::Gif,
            image::ImageFormat::WebP => Self::WebP,
            image::ImageFormat::Bmp => Self::Bmp,
            image::ImageFormat::Tiff => Self::Tiff,
            image::ImageFormat::Ico => Self::Ico,
            image::ImageFormat::Avif => Self::Avif,
            _ => return Err(()),
        })
    }
}

/// Structural metadata for a recognized image.
#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct ImageInfo {
    pub(crate) format: ImageFormat,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) animated: bool,
}

/// A model-ready, bounded raster image.
#[derive(Debug)]
pub(crate) struct PreparedImage {
    pub(crate) mime: &'static str,
    pub(crate) data: String,
}

/// Decode, bound, and encode an image for a model image-content block.
pub(crate) fn preprocess(bytes: &[u8]) -> Result<PreparedImage> {
    let image = image::load_from_memory(bytes).context("decode image")?;
    let image = image.resize(MAX_LONG_EDGE, MAX_LONG_EDGE, FilterType::Lanczos3);
    let (mime, bytes) = encode_image(&image)?;
    Ok(PreparedImage {
        mime,
        data: STANDARD.encode(bytes),
    })
}

fn encode_image(image: &DynamicImage) -> Result<(&'static str, Vec<u8>)> {
    let mut bytes = Vec::new();
    if image.color().has_alpha() {
        let mut cursor = Cursor::new(&mut bytes);
        image
            .write_to(&mut cursor, RasterFormat::Png)
            .context("encode PNG")?;
        Ok(("image/png", bytes))
    } else {
        JpegEncoder::new_with_quality(&mut bytes, JPEG_QUALITY)
            .encode_image(image)
            .context("encode JPEG")?;
        Ok(("image/jpeg", bytes))
    }
}

impl ImageInfo {
    /// Identify `bytes` as an image, reporting its format, pixel dimensions, and a
    /// best-effort animation flag. Returns `None` when the bytes are not a
    /// supported image.
    pub(crate) fn probe(bytes: &[u8]) -> Option<Self> {
        let reader = image::ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .ok()?;
        let format = ImageFormat::try_from(reader.format()?).ok()?;
        let (width, height) = reader.into_dimensions().ok()?;
        let animated = match format {
            ImageFormat::Png => png_animated(bytes),
            ImageFormat::Gif => gif_animated(bytes),
            ImageFormat::WebP => webp_animated(bytes),
            _ => false,
        };
        Some(ImageInfo {
            format,
            width: usize::try_from(width).ok()?,
            height: usize::try_from(height).ok()?,
            animated,
        })
    }
}

/// Whether a PNG carries an `acTL` chunk (APNG) ahead of its first `IDAT`.
fn png_animated(bytes: &[u8]) -> bool {
    let mut pos = 8;
    while pos + 8 <= bytes.len() {
        let len = u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
            as usize;
        let tag = &bytes[pos + 4..pos + 8];
        if tag == b"acTL" {
            return true;
        }
        if tag == b"IDAT" {
            return false;
        }
        pos = pos.saturating_add(12).saturating_add(len);
    }
    false
}

/// Whether a GIF contains more than one image frame.
fn gif_animated(bytes: &[u8]) -> bool {
    if bytes.len() < 13 {
        return false;
    }
    let screen_flags = bytes[10];
    let mut pos = 13;
    if screen_flags & 0x80 != 0 {
        pos += 3 * (1usize << ((screen_flags & 0x07) + 1));
    }
    let mut frames = 0u32;
    while let Some(&block) = bytes.get(pos) {
        match block {
            0x2c => {
                frames += 1;
                if frames > 1 {
                    return true;
                }
                let Some(&local_flags) = bytes.get(pos + 9) else {
                    return false;
                };
                pos += 10;
                if local_flags & 0x80 != 0 {
                    pos += 3 * (1usize << ((local_flags & 0x07) + 1));
                }
                pos = skip_sub_blocks(bytes, pos + 1);
            }
            0x21 => pos = skip_sub_blocks(bytes, pos + 2),
            _ => break,
        }
    }
    false
}

/// Whether a RIFF/WebP container declares animation via an `ANIM` chunk.
fn webp_animated(bytes: &[u8]) -> bool {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WEBP" {
        return false;
    }
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        if &bytes[pos..pos + 4] == b"ANIM" {
            return true;
        }
        let size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        pos = pos
            .saturating_add(8)
            .saturating_add(size)
            .saturating_add(size & 1);
    }
    false
}

/// Advance past a GIF sub-block chain, returning the offset after the
/// terminating zero-length block.
fn skip_sub_blocks(bytes: &[u8], mut pos: usize) -> usize {
    while let Some(&size) = bytes.get(pos) {
        pos += 1;
        if size == 0 {
            break;
        }
        pos += usize::from(size);
    }
    pos
}
