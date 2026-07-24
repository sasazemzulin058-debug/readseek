// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

use crate::engine::hash::{LineHash, hash_line, hash_text};
use crate::engine::image::ImageInfo;
use crate::engine::lang::{
    AnalysisEngine, DocumentKind, Language, detect_by_path, detect_language, normalize_source_text,
};
use crate::engine::paths::bytes_contain_identifier;
use crate::engine::pdf::PdfInfo;
use crate::engine::symbols;
use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub(crate) enum ContentCategory {
    Text,
    Image(ImageInfo),
    Pdf(PdfInfo),
    Binary,
}

#[derive(Debug)]
pub(crate) struct Detection {
    pub(crate) file: PathBuf,
    pub(crate) language: Language,
    pub(crate) engine: Option<AnalysisEngine>,
    pub(crate) supported: bool,
    pub(crate) category: ContentCategory,
    pub(crate) mime: Option<String>,
    pub(crate) syntax: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct HashLine {
    pub(crate) line: usize,
    pub(crate) hash: LineHash,
    pub(crate) text: String,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Symbol {
    pub(crate) kind: String,
    pub(crate) name: String,
    pub(crate) qualified_name: String,
    pub(crate) start_line: usize,
    pub(crate) end_line: usize,
    pub(crate) start_hash: LineHash,
    pub(crate) end_hash: LineHash,
    pub(crate) start_byte: usize,
    pub(crate) end_byte: usize,
    pub(crate) name_byte: usize,
}

#[derive(Debug)]
pub(crate) struct SourceFile {
    pub(crate) path: PathBuf,
    pub(crate) text: String,
    pub(crate) kind: DocumentKind,
    pub(crate) detection: Detection,
    pub(crate) lines: Vec<SourceLine>,
    pub(crate) line_starts: Vec<usize>,
    pub(crate) file_hash: String,
    pub(crate) document_bytes: Option<Vec<u8>>,
}

impl SourceFile {
    pub(crate) fn line(&self, n: usize) -> Option<&SourceLine> {
        self.lines.get(n.checked_sub(1)?)
    }

    /// Zero-based index into `lines`/`line_starts` of the line holding `byte`.
    pub(crate) fn line_index(&self, byte: usize) -> usize {
        self.line_starts
            .partition_point(|&start| start <= byte)
            .saturating_sub(1)
    }

    /// One-based `(line, column)` of `byte`, with `column` measured in bytes.
    pub(crate) fn line_column(&self, byte: usize) -> (usize, usize) {
        if self.line_starts.is_empty() {
            return (1, 1);
        }
        let index = self.line_index(byte);
        let number = self.lines.get(index).map_or(1, |line| line.number);
        (number, byte - self.line_starts[index] + 1)
    }

    pub(crate) fn cursor_byte(&self, line: usize, column: usize) -> Result<usize> {
        if line == 0 || column == 0 {
            bail!("line and column must be greater than zero");
        }
        let source_line = self
            .line(line)
            .with_context(|| format!("line {line} not found in {}", self.path.display()))?;
        let max_column = source_line.text.len() + 1;
        if column > max_column {
            bail!("column {column} exceeds maximum column {max_column} for line {line}");
        }
        Ok(self.line_starts[line - 1] + column - 1)
    }

    /// Reject non-text documents (images, binaries) before source processing.
    pub(crate) fn require_text(&self) -> Result<()> {
        if matches!(self.detection.category, ContentCategory::Text) {
            return Ok(());
        }
        bail!(
            "unsupported binary file: {} ({})",
            self.path.display(),
            self.detection.mime.as_deref().unwrap_or("unknown mime")
        );
    }
}

#[derive(Debug)]
struct LoadedDocument {
    text: String,
    category: ContentCategory,
    mime: Option<String>,
    document_bytes: Option<Vec<u8>>,
}

#[derive(Debug)]
pub(crate) struct SourceLine {
    pub(crate) number: usize,
    pub(crate) text: String,
}

impl SourceLine {
    pub(crate) fn hash(&self) -> LineHash {
        hash_line(&self.text)
    }
}

impl From<&SourceLine> for HashLine {
    fn from(line: &SourceLine) -> Self {
        Self {
            line: line.number,
            hash: line.hash(),
            text: line.text.clone(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct SourceMap {
    pub(crate) symbols: Vec<Symbol>,
}

pub(crate) fn load_source(path: &Path, language: Option<Language>) -> Result<SourceFile> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    load_source_from_bytes(path, bytes, language)
}

pub(crate) fn load_source_from_bytes(
    path: &Path,
    bytes: Vec<u8>,
    language: Option<Language>,
) -> Result<SourceFile> {
    let document = load_document(path, bytes)?;
    let mut source = source_from_text(
        path,
        document.text,
        language,
        document.category,
        document.mime,
    );
    source.document_bytes = document.document_bytes;
    Ok(source)
}

/// Load `path` as a source file, but only when it contains `name` as a whole
/// identifier.
///
/// Returns `None` on a read, UTF-8, or parse error, or when the identifier is
/// absent, so callers scanning many files can skip it without distinguishing the
/// reasons.
pub(crate) fn read_source_containing(
    path: &Path,
    name: &str,
    language: Option<Language>,
) -> Option<SourceFile> {
    read_source_containing_with_buffer(path, name, language, &mut Vec::new())
}

pub(crate) fn read_source_containing_with_buffer(
    path: &Path,
    name: &str,
    language: Option<Language>,
    bytes: &mut Vec<u8>,
) -> Option<SourceFile> {
    bytes.clear();
    File::open(path).ok()?.read_to_end(bytes).ok()?;
    if !bytes_contain_identifier(bytes, name.as_bytes()) {
        return None;
    }
    let text = String::from_utf8(std::mem::take(bytes)).ok()?;
    Some(source_from_text(
        path,
        text,
        language,
        ContentCategory::Text,
        None,
    ))
}

fn detect_mime(bytes: &[u8]) -> Option<String> {
    if bytes.starts_with(b"%PDF-") {
        return Some("application/pdf".to_owned());
    }
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some("image/png".to_owned());
    }
    if bytes.starts_with(b"\xff\xd8\xff") {
        return Some("image/jpeg".to_owned());
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some("image/gif".to_owned());
    }
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return Some("image/webp".to_owned());
    }
    if bytes.starts_with(b"BM") {
        return Some("image/bmp".to_owned());
    }
    if bytes.starts_with(b"II*\x00") || bytes.starts_with(b"MM\x00*") {
        return Some("image/tiff".to_owned());
    }
    if bytes.starts_with(b"#!") {
        return Some("text/x-shellscript".to_owned());
    }

    let mut start = 0;
    if bytes.starts_with(b"\xef\xbb\xbf") {
        start += 3;
    }
    while start < bytes.len() && bytes[start].is_ascii_whitespace() {
        start += 1;
    }
    let rest = &bytes[start..];

    if rest.starts_with(b"<?xml") || rest.starts_with(b"<?XML") {
        return Some("text/xml".to_owned());
    }
    if rest.len() >= 14 && rest[..14].eq_ignore_ascii_case(b"<!doctype html") {
        return Some("text/html".to_owned());
    }
    if rest.len() >= 5 && rest[..5].eq_ignore_ascii_case(b"<html") {
        return Some("text/html".to_owned());
    }

    None
}

pub(crate) fn load_indexable_source(
    path: &Path,
    language: Option<Language>,
) -> Result<Option<SourceFile>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mime = detect_mime(&bytes);
    if !is_plaintext(&bytes, mime.as_deref()) {
        return Ok(None);
    }
    let Ok(text) = String::from_utf8(bytes) else {
        return Ok(None);
    };
    Ok(Some(source_from_text(
        path,
        text,
        language,
        ContentCategory::Text,
        mime,
    )))
}

pub(crate) fn source_from_text(
    path: &Path,
    text: String,
    language: Option<Language>,
    category: ContentCategory,
    mime: Option<String>,
) -> SourceFile {
    let text = normalize_source_text(text);
    let path_language = detect_by_path(path);
    let (detected_language, syntax) = if !matches!(category, ContentCategory::Text)
        && language.is_none()
        && path_language.is_none()
    {
        (Language::Unknown, None)
    } else {
        detect_language(path, &text)
    };
    let language = language.unwrap_or(detected_language);
    let engine = language.engine();
    let kind = if language == Language::Unknown {
        DocumentKind::Text
    } else {
        DocumentKind::Source
    };
    let file_hash = hash_text(&text);
    let line_count = memchr::memchr_iter(b'\n', text.as_bytes()).count() + 1;
    let mut line_starts = Vec::with_capacity(line_count);
    let mut lines = Vec::with_capacity(line_count);
    let mut offset = 0;
    for (index, part) in text.split_terminator('\n').enumerate() {
        line_starts.push(offset);
        offset += part.len() + 1;
        lines.push(SourceLine {
            number: index + 1,
            text: part.to_owned(),
        });
    }

    SourceFile {
        path: path.to_path_buf(),
        text,
        kind,
        detection: Detection {
            file: path.to_path_buf(),
            language,
            engine,
            supported: language != Language::Unknown,
            category,
            mime,
            syntax,
        },
        lines,
        line_starts,
        file_hash,
        document_bytes: None,
    }
}

fn load_document(path: &Path, bytes: Vec<u8>) -> Result<LoadedDocument> {
    let mut mime = detect_mime(&bytes);

    let (category, text, document_bytes) = if is_plaintext(&bytes, mime.as_deref()) {
        let text = String::from_utf8(bytes)
            .with_context(|| format!("{} is not UTF-8 text", path.display()))?;
        (ContentCategory::Text, text, None)
    } else if let Some(image) = crate::engine::image::probe(&bytes) {
        if mime.is_none() {
            mime = Some(image.format.mime_type().to_owned());
        }
        (ContentCategory::Image(image), String::new(), Some(bytes))
    } else if mime.as_deref() == Some("application/pdf") || bytes.starts_with(b"%PDF-") {
        let pdf = crate::engine::pdf::probe(&bytes)?;
        (ContentCategory::Pdf(pdf), String::new(), Some(bytes))
    } else {
        (ContentCategory::Binary, String::new(), None)
    };

    Ok(LoadedDocument {
        text,
        category,
        mime,
        document_bytes,
    })
}

fn is_plaintext(bytes: &[u8], mime: Option<&str>) -> bool {
    let text_mime = mime.is_none_or(|m| m.starts_with("text/"));
    text_mime && !bytes.contains(&0)
}

pub(crate) fn source_map(source: &SourceFile) -> Result<SourceMap> {
    let readseek_dir = crate::engine::repo::find_readseek_dir(&source.path);
    source_map_with_dir(source, readseek_dir.as_deref())
}

pub(crate) fn source_map_with_dir(
    source: &SourceFile,
    readseek_dir: Option<&Path>,
) -> Result<SourceMap> {
    if let Some(readseek_dir) = readseek_dir {
        match crate::engine::repo::load_map(readseek_dir, &source.file_hash) {
            Ok(Some((source_map, language, _engine))) => {
                if language == source.detection.language {
                    return Ok(source_map);
                }
                tracing::debug!(
                    target: "tracing",
                    "cache language mismatch for {}: cached {language}, current {}",
                    source.path.display(),
                    source.detection.language
                );
            }
            Ok(None) => {}
            Err(error) => tracing::debug!(target: "tracing", "cache load error: {error:#}"),
        }

        let source_map = symbols::parse_source_map(source)?;
        if let Err(error) =
            crate::engine::repo::store_map(readseek_dir, &source.file_hash, source, &source_map)
        {
            tracing::debug!(target: "tracing", "cache store error: {error:#}");
        }

        return Ok(source_map);
    }

    symbols::parse_source_map(source)
}

pub(crate) fn find_symbol(source_map: &SourceMap, line: usize) -> Option<Symbol> {
    let symbols = &source_map.symbols;
    let idx = symbols.partition_point(|s| s.start_line <= line);
    (0..idx)
        .rev()
        .filter(|&i| symbols[i].end_line >= line)
        .min_by_key(|&i| symbols[i].end_line - symbols[i].start_line)
        .map(|i| symbols[i].clone())
}

pub(crate) fn line_hash(source: &SourceFile, line: usize) -> Result<LineHash> {
    source
        .line(line)
        .map(SourceLine::hash)
        .with_context(|| format!("line {line} not found in {}", source.path.display()))
}

pub(crate) fn range_hashlines(
    source: &SourceFile,
    start_line: usize,
    end_line: usize,
) -> Vec<HashLine> {
    let start = start_line.saturating_sub(1);
    let end = end_line.min(source.lines.len());
    source.lines[start..end]
        .iter()
        .map(HashLine::from)
        .collect()
}
