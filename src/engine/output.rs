// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

use std::io::{self, Read as _};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use strum_macros::{AsRefStr, Display};

use crate::engine::hash::LineHash;
use crate::engine::image::{ImageInfo, PreparedImage};
use crate::engine::lang::{EngineField, Language};
use crate::engine::source::{
    ContentCategory, Detection, HashLine, SourceFile, Symbol, find_symbol, load_source,
    load_source_from_bytes, range_hashlines, source_map,
};
use crate::engine::target::{Target, TargetAddress};
use crate::engine::vision::{Analysis, DetectedObject};
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Display, AsRefStr)]
#[strum(serialize_all = "lowercase")]
pub(crate) enum Format {
    #[default]
    Json,
    Plain,
}

impl FromStr for Format {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "json" => Ok(Self::Json),
            "plain" => Ok(Self::Plain),
            _ => Err(format!("invalid format '{value}': expected json or plain")),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum DetectOutput {
    Source(DetectSourceOutput),
    Image(DetectImageOutput),
    Pdf(DetectPdfOutput),
    Binary(DetectBinaryOutput),
    Text(DetectTextOutput),
}

impl DetectOutput {
    pub(crate) fn from_detection(detection: Detection) -> Self {
        let Detection {
            file,
            language,
            engine,
            supported,
            category,
            mime: detect_mime,
            syntax,
        } = detection;
        let mime = detect_mime
            .clone()
            .unwrap_or_else(|| "text/plain".to_string());

        match category {
            ContentCategory::Image(image) => {
                Self::Image(DetectImageOutput::new(file, mime, detect_mime, image))
            }
            ContentCategory::Pdf(pdf) => Self::Pdf(DetectPdfOutput {
                type_: mime,
                file,
                mime: detect_mime,
                format: "pdf",
                pages: pdf.pages,
            }),
            ContentCategory::Binary => Self::Binary(DetectBinaryOutput {
                file,
                type_: mime,
                mime: detect_mime,
            }),
            ContentCategory::Text => {
                if language == Language::Unknown {
                    Self::Text(DetectTextOutput {
                        file,
                        type_: mime,
                        mime: detect_mime,
                    })
                } else {
                    Self::Source(DetectSourceOutput {
                        file,
                        language,
                        engine: EngineField(engine),
                        supported,
                        type_: mime,
                        mime: detect_mime,
                        syntax,
                    })
                }
            }
        }
    }
}
#[derive(Debug, Serialize)]
pub(crate) struct DetectSourceOutput {
    #[serde(rename = "type")]
    type_: String,
    file: PathBuf,
    language: Language,
    #[serde(skip_serializing_if = "EngineField::is_none")]
    engine: EngineField,
    supported: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    mime: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    syntax: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct DetectImageOutput {
    #[serde(rename = "type")]
    type_: String,
    file: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    mime: Option<String>,
    #[serde(flatten)]
    image: ImageInfo,
}

impl DetectImageOutput {
    fn new(file: PathBuf, type_: String, mime: Option<String>, image: ImageInfo) -> Self {
        Self {
            type_,
            file,
            mime,
            image,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct DetectBinaryOutput {
    #[serde(rename = "type")]
    type_: String,
    file: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    mime: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct DetectPdfOutput {
    #[serde(rename = "type")]
    type_: String,
    file: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    mime: Option<String>,
    format: &'static str,
    pages: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct DetectTextOutput {
    #[serde(rename = "type")]
    type_: String,
    file: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    mime: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct SourceHeader {
    file: PathBuf,
    language: Language,
    #[serde(skip_serializing_if = "EngineField::is_none")]
    engine: EngineField,
    line_count: usize,
    file_hash: String,
}

impl From<&SourceFile> for SourceHeader {
    fn from(source: &SourceFile) -> Self {
        Self {
            file: source.path.clone(),
            language: source.detection.language,
            engine: EngineField(source.detection.engine),
            line_count: source.lines.len(),
            file_hash: source.file_hash.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ImageMode {
    #[default]
    None,
    All,
    Caption,
    Objects,
    Ocr,
}

impl FromStr for ImageMode {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "none" => Ok(Self::None),
            "all" => Ok(Self::All),
            "caption" => Ok(Self::Caption),
            "objects" => Ok(Self::Objects),
            "ocr" => Ok(Self::Ocr),
            _ => Err(format!(
                "unknown vision mode `{value}`; expected none, all, caption, objects, or ocr"
            )),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum ReadOutput {
    Text(ReadTextOutput),
    Image(ReadImageOutput),
    Pdf(crate::engine::pdf::ReadPdfOutput),
}

#[derive(Debug, Serialize)]
pub(crate) struct ReadTextOutput {
    #[serde(flatten)]
    header: SourceHeader,
    start_line: usize,
    end_line: usize,
    hashlines: Vec<HashLine>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ReadImageOutput {
    file: PathBuf,
    #[serde(rename = "type")]
    type_: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    mime: Option<String>,
    #[serde(flatten)]
    image: ImageInfo,
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

#[derive(Debug, Serialize)]
pub(crate) struct MapOutput {
    #[serde(flatten)]
    header: SourceHeader,
    symbols: Vec<Symbol>,
}

#[derive(Debug, Serialize)]
pub(crate) struct CheckOutput {
    #[serde(flatten)]
    header: SourceHeader,
    error_count: usize,
    missing_count: usize,
    diagnostics: Vec<Diagnostic>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum DiagnosticKind {
    Error,
    Missing,
}

#[derive(Debug, Serialize)]
pub(crate) struct Diagnostic {
    pub(crate) kind: DiagnosticKind,
    pub(crate) start_line: usize,
    pub(crate) end_line: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct SymbolOutput {
    #[serde(flatten)]
    header: SourceHeader,
    symbol: Symbol,
    hashlines: Vec<HashLine>,
}

#[derive(Debug, Serialize)]
pub(crate) struct IdentifyOutput {
    #[serde(flatten)]
    header: SourceHeader,
    line: usize,
    column: usize,
    line_hash: LineHash,
    hashlines: Vec<HashLine>,
    #[serde(skip_serializing_if = "Option::is_none")]
    identifier: Option<IdentifierOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol: Option<Symbol>,
}

#[derive(Debug, Serialize)]
pub(crate) struct IdentifierOutput {
    text: String,
    start_column: usize,
    end_column: usize,
    start_byte: usize,
    end_byte: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct DefOutput {
    pub(crate) definitions: Vec<DefLocation>,
}

#[derive(Debug, Serialize)]
pub(crate) struct DefLocation {
    pub(crate) file: PathBuf,
    pub(crate) language: Language,
    #[serde(skip_serializing_if = "EngineField::is_none")]
    pub(crate) engine: EngineField,
    pub(crate) file_hash: String,
    pub(crate) symbol: Symbol,
    #[serde(skip_serializing)]
    pub(crate) line_hash: LineHash,
    #[serde(skip_serializing)]
    pub(crate) text: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct RefsOutput {
    pub(crate) references: Vec<RefLocation>,
}

#[derive(Debug, Serialize)]
pub(crate) struct RefLocation {
    pub(crate) file: Arc<PathBuf>,
    pub(crate) language: Language,
    #[serde(skip_serializing_if = "EngineField::is_none")]
    pub(crate) engine: EngineField,
    pub(crate) file_hash: Arc<str>,
    pub(crate) line: usize,
    pub(crate) column: usize,
    pub(crate) line_hash: LineHash,
    pub(crate) text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) symbol: Option<Symbol>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) occurrence: Option<crate::engine::binding::OccurrenceKind>,
}

#[derive(Debug, Serialize)]
pub(crate) struct CompactOutput {
    pub(crate) locations: Vec<CompactLocation>,
}

#[derive(Debug, Serialize)]
pub(crate) struct CompactLocation {
    pub(crate) file: Arc<PathBuf>,
    pub(crate) line: usize,
    pub(crate) column: usize,
    pub(crate) line_hash: LineHash,
    pub(crate) text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) qualified_name: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct RenameOutput {
    pub(crate) file: PathBuf,
    pub(crate) language: Language,
    #[serde(skip_serializing_if = "EngineField::is_none")]
    pub(crate) engine: EngineField,
    pub(crate) file_hash: String,
    pub(crate) old_name: String,
    pub(crate) new_name: String,
    pub(crate) plan_hash: String,
    pub(crate) applied: bool,
    pub(crate) conflicts: Vec<RenameConflict>,
    pub(crate) edits: Vec<RenameEdit>,
    /// Additional files edited when `--workspace` expands the rename. The
    /// top-level fields describe the cursor file (binding-accurate); each entry
    /// here is matched by name (binding-resolved only to drop local shadows).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) others: Vec<RenameFileOutput>,
}

#[derive(Debug, Serialize)]
pub(crate) struct RenameFileOutput {
    pub(crate) file: PathBuf,
    pub(crate) language: Language,
    #[serde(skip_serializing_if = "EngineField::is_none")]
    pub(crate) engine: EngineField,
    pub(crate) file_hash: String,
    pub(crate) conflicts: Vec<RenameConflict>,
    pub(crate) edits: Vec<RenameEdit>,
}

#[derive(Debug, Serialize)]
pub(crate) struct RenameEdit {
    pub(crate) line: usize,
    pub(crate) start_column: usize,
    pub(crate) end_column: usize,
    pub(crate) start_byte: usize,
    pub(crate) end_byte: usize,
    pub(crate) occurrence: crate::engine::binding::OccurrenceKind,
    pub(crate) line_hash: LineHash,
    pub(crate) text: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct RenameConflict {
    pub(crate) line: usize,
    pub(crate) column: usize,
    pub(crate) reason: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct SearchOutput {
    pub(crate) results: Vec<SearchFileOutput>,
}

#[derive(Debug, Serialize)]
pub(crate) struct SearchFileOutput {
    pub(crate) file: PathBuf,
    pub(crate) language: Language,
    #[serde(skip_serializing_if = "EngineField::is_none")]
    pub(crate) engine: EngineField,
    pub(crate) file_hash: String,
    pub(crate) matches: Vec<SearchMatch>,
}

#[derive(Debug, Serialize)]
pub(crate) struct SearchMatch {
    pub(crate) start_line: usize,
    pub(crate) end_line: usize,
    pub(crate) start_hash: LineHash,
    pub(crate) end_hash: LineHash,
    pub(crate) hashlines: Vec<HashLine>,
    pub(crate) captures: Vec<SearchCapture>,
}

#[derive(Debug, Serialize)]
pub(crate) struct SearchCapture {
    pub(crate) name: String,
    pub(crate) start_line: usize,
    pub(crate) end_line: usize,
    pub(crate) start_hash: LineHash,
    pub(crate) end_hash: LineHash,
    pub(crate) hashlines: Vec<HashLine>,
}

pub(crate) fn resolve_target(source: &SourceFile, target: &Target) -> Result<Option<usize>> {
    match target.address.as_ref() {
        Some(TargetAddress::Line(line)) => Ok(Some(*line)),
        Some(TargetAddress::Hash(hash)) => {
            let target = hash
                .parse::<LineHash>()
                .with_context(|| format!("invalid target hash {hash}"))?;
            source
                .lines
                .iter()
                .find_map(|line| (line.hash() == target).then_some(line.number))
                .with_context(|| format!("hash {hash} not found in {}", source.path.display()))
                .map(Some)
        }
        Some(TargetAddress::Name(_)) => {
            bail!("name targets are not supported for this command")
        }
        None => Ok(None),
    }
}

pub(crate) fn load_source_for_input(
    target: &Target,
    override_language: Option<Language>,
) -> Result<SourceFile> {
    if target.read_stdin {
        let mut bytes = Vec::new();
        io::stdin().read_to_end(&mut bytes).context("read stdin")?;
        let path = if target.path.as_os_str().is_empty() {
            Path::new("<stdin>")
        } else {
            &target.path
        };
        return load_source_from_bytes(path, bytes, override_language);
    }
    load_source(&target.path, override_language)
}

pub(crate) fn read_output(
    source: &SourceFile,
    start: Option<usize>,
    end: Option<usize>,
) -> Result<ReadTextOutput> {
    let line_count = source.lines.len();
    let start_line = start.unwrap_or(1);
    let requested_end_line = end.unwrap_or(line_count);
    let end_line = requested_end_line.min(line_count);

    if start_line == 0 {
        bail!("start line must be greater than zero");
    }
    if line_count == 0 && start.is_none() && end.is_none() {
        return Ok(ReadTextOutput {
            header: source.into(),
            start_line,
            end_line,
            hashlines: Vec::new(),
        });
    }
    if requested_end_line < start_line {
        bail!("end line must be greater than or equal to start line");
    }
    if start_line > line_count {
        bail!("start line {start_line} exceeds line count {line_count}");
    }
    let slice_start = start_line - 1;

    let hashlines = source.lines[slice_start..end_line]
        .iter()
        .map(HashLine::from)
        .collect();

    Ok(ReadTextOutput {
        header: source.into(),
        start_line,
        end_line,
        hashlines,
    })
}

pub(crate) fn read_image_output(
    source: &SourceFile,
    mode: ImageMode,
    analysis: Analysis,
    prepared: Option<PreparedImage>,
) -> Result<ReadImageOutput> {
    let mime = source.detection.mime.clone();
    let ContentCategory::Image(image) = source.detection.category else {
        bail!("not an image");
    };
    let (caption, objects, ocr) = match mode {
        ImageMode::None => (None, None, None),
        ImageMode::All => (analysis.caption, analysis.objects, analysis.ocr),
        ImageMode::Caption => (analysis.caption, None, None),
        ImageMode::Objects => (None, analysis.objects, None),
        ImageMode::Ocr => (None, None, analysis.ocr),
    };
    let (type_, mime, encoding, data) = if let Some(prepared) = prepared {
        (
            prepared.mime.to_owned(),
            Some(prepared.mime.to_owned()),
            Some("base64"),
            Some(prepared.data),
        )
    } else {
        (
            mime.clone()
                .unwrap_or_else(|| "application/octet-stream".to_string()),
            mime,
            None,
            None,
        )
    };
    Ok(ReadImageOutput {
        file: source.path.clone(),
        type_,
        mime,
        image,
        mode,
        caption,
        objects,
        ocr,
        encoding,
        data,
    })
}

pub(crate) fn map_output(source: &SourceFile) -> Result<MapOutput> {
    let source_map = source_map(source)?;

    Ok(MapOutput {
        header: source.into(),
        symbols: source_map.symbols,
    })
}

pub(crate) fn check_output(source: &SourceFile) -> Result<CheckOutput> {
    let diagnostics = crate::engine::symbols::parse_diagnostics(source)?;
    let error_count = diagnostics
        .iter()
        .filter(|diagnostic| matches!(diagnostic.kind, DiagnosticKind::Error))
        .count();
    let missing_count = diagnostics.len() - error_count;

    Ok(CheckOutput {
        header: source.into(),
        error_count,
        missing_count,
        diagnostics,
    })
}

/// How [`symbol_output`] locates the symbol to report.
#[derive(Clone, Copy)]
pub(crate) enum SymbolAddress<'a> {
    /// A qualified or unqualified symbol name.
    Name(&'a str),
    /// A one-based line the symbol must enclose.
    Line(usize),
}

pub(crate) fn symbol_output(
    source: &SourceFile,
    address: SymbolAddress<'_>,
) -> Result<SymbolOutput> {
    let source_map = source_map(source)?;
    let symbol = match address {
        SymbolAddress::Name(address) => {
            let mut matches = source_map
                .symbols
                .iter()
                .filter(|symbol| symbol.qualified_name == address || symbol.name == address);

            let symbol = matches
                .next()
                .with_context(|| format!("no symbol `{address}`"))?;

            if matches.next().is_some() {
                bail!("ambiguous symbol `{address}`");
            }

            symbol.clone()
        }
        SymbolAddress::Line(line) => {
            find_symbol(&source_map, line).with_context(|| format!("no symbol at line {line}"))?
        }
    };

    Ok(SymbolOutput {
        header: source.into(),
        hashlines: range_hashlines(source, symbol.start_line, symbol.end_line),
        symbol,
    })
}

pub(crate) fn identify_output(
    source: &SourceFile,
    target_line: Option<usize>,
    column: Option<usize>,
) -> Result<IdentifyOutput> {
    let line = target_line.context("identify requires a target line or hash")?;
    let column = column.unwrap_or(1);
    let cursor_byte = source.cursor_byte(line, column)?;
    let source_line = source
        .line(line)
        .with_context(|| format!("line {line} not found in {}", source.path.display()))?;
    let line_start = source.line_starts[line - 1];
    let identifier = crate::engine::symbols::token_at(source, cursor_byte)
        .map(|token| IdentifierOutput {
            text: token.text,
            start_column: token.start_byte - line_start + 1,
            end_column: token.end_byte - line_start + 1,
            start_byte: token.start_byte,
            end_byte: token.end_byte,
        })
        .or_else(|| identify_byte_scan(source_line, line_start, column));
    let source_map = source_map(source)?;
    let symbol = find_symbol(&source_map, line);

    Ok(IdentifyOutput {
        header: source.into(),
        line,
        column,
        line_hash: source_line.hash(),
        hashlines: vec![HashLine::from(source_line)],
        identifier,
        symbol,
    })
}

/// Fallback identifier extraction for languages without a tree-sitter parser.
///
/// Walks ASCII identifier bytes around the cursor on a single line. `line_start`
/// is the byte offset of the line within the file, used to report absolute bytes.
fn identify_byte_scan(
    source_line: &crate::engine::source::SourceLine,
    line_start: usize,
    column: usize,
) -> Option<IdentifierOutput> {
    let bytes = source_line.text.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let mut index = column.saturating_sub(1).min(bytes.len().saturating_sub(1));
    if !(bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
        && index > 0
        && (bytes[index - 1].is_ascii_alphanumeric() || bytes[index - 1] == b'_')
    {
        index -= 1;
    }
    if !(bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_') {
        return None;
    }
    let mut start = index;
    while start > 0 && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_') {
        start -= 1;
    }
    let mut end = index + 1;
    while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') {
        end += 1;
    }
    Some(IdentifierOutput {
        text: source_line.text[start..end].to_owned(),
        start_column: start + 1,
        end_column: end + 1,
        start_byte: line_start + start,
        end_byte: line_start + end,
    })
}
