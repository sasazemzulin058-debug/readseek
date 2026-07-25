// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

use crate::engine::flags::GitFlags;
use crate::engine::hash::LineHash;
use crate::engine::lang::{AnalysisEngine, EngineField, Language};
use crate::engine::output::{CompactLocation, CompactOutput, RefLocation, RefsOutput};
use crate::engine::paths::{command_paths, identifier_spans};
use crate::engine::source::{
    ContentCategory, SourceFile, Symbol, read_source_containing_with_buffer, source_from_text,
    source_map_with_dir,
};
use crate::engine::symbols;
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tree_sitter::{Node, Parser};

/// Inputs for [`output`]: the identifier to find and the search scope.
pub(crate) struct Request {
    pub(crate) target: PathBuf,
    pub(crate) name: String,
    pub(crate) scope: bool,
    pub(crate) line: Option<usize>,
    pub(crate) column: Option<usize>,
    pub(crate) language: Option<Language>,
    pub(crate) flags: GitFlags,
}

pub(crate) fn output(request: &Request) -> Result<RefsOutput> {
    let name = &request.name;
    if name.is_empty() {
        bail!("reference name must not be empty");
    }
    if !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
        bail!("reference name must be an ASCII identifier");
    }
    if request.scope {
        return scoped_output(request);
    }
    if request.line.is_some() || request.column.is_some() {
        bail!("--line and --column require --scope");
    }
    let readseek_dir = crate::engine::repo::find_readseek_dir(&request.target);
    let paths = command_paths(&request.target, request.flags)?;

    let references: Vec<RefLocation> = paths
        .par_iter()
        .map_init(
            || (Parser::new(), Vec::new()),
            |(parser, bytes), path| {
                let Some(source) =
                    read_source_containing_with_buffer(path, name, request.language, bytes)
                else {
                    return vec![];
                };
                let needs_parser = matches!(
                    source.detection.language,
                    crate::engine::lang::Language::C | crate::engine::lang::Language::Cpp
                );
                let parser = needs_parser.then_some(&mut *parser);
                scan_source(&source, name, parser, readseek_dir.as_deref())
            },
        )
        .flatten_iter()
        .collect();

    Ok(RefsOutput { references })
}

/// Resolve a single binding within one file and classify its occurrences.
fn scoped_output(request: &Request) -> Result<RefsOutput> {
    let line = request.line.context("--scope requires --line")?;
    let column = request.column.unwrap_or(1);
    if !request.target.is_file() {
        bail!("--scope requires a single regular file target");
    }
    let bytes =
        fs::read(&request.target).with_context(|| format!("read {}", request.target.display()))?;
    let text = String::from_utf8(bytes).context("file is not valid UTF-8")?;
    let source = source_from_text(
        &request.target,
        text,
        request.language,
        ContentCategory::Text,
        None,
    );
    let cursor_byte = source.cursor_byte(line, column)?;
    let binding = crate::engine::binding::resolve(&source, cursor_byte).with_context(|| {
        format!(
            "no resolvable binding at {}:{line}:{column}",
            request.target.display()
        )
    })?;
    if binding.name != request.name {
        bail!(
            "binding at cursor is `{}`, not `{}`",
            binding.name,
            request.name
        );
    }

    let source_map = source_map_with_dir(&source, None).ok();
    let file = Arc::new(source.path.clone());
    let file_hash: Arc<str> = Arc::from(source.file_hash.as_str());
    let references = binding
        .occurrences
        .iter()
        .filter(|occurrence| occurrence.kind != crate::engine::binding::OccurrenceKind::Shadowed)
        .map(|occurrence| {
            let line_idx = source.line_index(occurrence.start_byte);
            let source_line = &source.lines[line_idx];
            RefLocation {
                file: Arc::clone(&file),
                language: source.detection.language,
                engine: EngineField(source.detection.engine),
                file_hash: Arc::clone(&file_hash),
                line: source_line.number,
                column: occurrence.start_byte - source.line_starts[line_idx] + 1,
                line_hash: source_line.hash(),
                text: source_line.text.clone(),
                symbol: source_map
                    .as_ref()
                    .and_then(|sm| sm.find_symbol(source_line.number).cloned()),
                occurrence: Some(occurrence.kind),
            }
        })
        .collect();

    Ok(RefsOutput { references })
}

pub(crate) fn compact(output: &RefsOutput) -> CompactOutput {
    CompactOutput {
        locations: output
            .references
            .iter()
            .map(|reference| {
                let symbol = reference.symbol.as_ref();
                CompactLocation {
                    file: reference.file.clone(),
                    line: reference.line,
                    column: reference.column,
                    line_hash: reference.line_hash,
                    text: reference.text.clone(),
                    kind: symbol.map(|symbol| symbol.kind.clone()),
                    name: symbol.map(|symbol| symbol.name.clone()),
                    qualified_name: symbol.map(|symbol| symbol.qualified_name.clone()),
                }
            })
            .collect(),
    }
}

fn scan_source(
    source: &SourceFile,
    name: &str,
    parser: Option<&mut Parser>,
    readseek_dir: Option<&Path>,
) -> Vec<RefLocation> {
    let source_map = source_map_with_dir(source, readseek_dir).ok();
    let ignored_ranges = parser
        .map(|p| scan_ignored_ranges(source, p))
        .unwrap_or_default();
    let line_starts = &source.line_starts;

    let text_bytes = source.text.as_bytes();
    let name_bytes = name.as_bytes();

    let mut compact: Vec<(usize, usize)> = Vec::new();
    let mut line_idx = 0;
    let mut ignored_idx = 0;
    for byte_index in identifier_spans(text_bytes, name_bytes) {
        while line_idx + 1 < line_starts.len() && line_starts[line_idx + 1] <= byte_index {
            line_idx += 1;
        }
        let Some(line) = source.lines.get(line_idx) else {
            continue;
        };
        while ignored_idx < ignored_ranges.len() && ignored_ranges[ignored_idx].1 <= byte_index {
            ignored_idx += 1;
        }
        if ignored_ranges
            .get(ignored_idx)
            .is_some_and(|&(start, end)| start <= byte_index && byte_index < end)
        {
            continue;
        }
        compact.push((line.number, byte_index - line_starts[line_idx] + 1));
    }

    if compact.is_empty() {
        return Vec::new();
    }

    let file = Arc::new(source.path.clone());
    let file_hash: Arc<str> = Arc::from(source.file_hash.as_str());
    let language = source.detection.language;
    let engine = source.detection.engine;

    let mut references = Vec::with_capacity(compact.len());
    let mut last_line = 0;
    let mut cached_line_hash = LineHash::default();
    let mut cached_text = String::new();
    let mut cached_symbol: Option<Symbol> = None;

    for (line_number, column) in compact {
        if line_number != last_line {
            let line = &source.lines[line_number - 1];
            last_line = line_number;
            cached_line_hash = line.hash();
            cached_text.clone_from(&line.text);
            cached_symbol = source_map
                .as_ref()
                .and_then(|sm| sm.find_symbol(line_number).cloned());
        }

        references.push(RefLocation {
            file: Arc::clone(&file),
            language,
            engine: EngineField(engine),
            file_hash: Arc::clone(&file_hash),
            line: line_number,
            column,
            line_hash: cached_line_hash,
            text: cached_text.clone(),
            symbol: cached_symbol.clone(),
            occurrence: None,
        });
    }

    references
}

fn scan_ignored_ranges(source: &SourceFile, parser: &mut Parser) -> Vec<(usize, usize)> {
    if !matches!(source.detection.language, Language::C | Language::Cpp) {
        return Vec::new();
    }
    if source.detection.engine != Some(AnalysisEngine::TreeSitter) {
        return Vec::new();
    }
    let Some(language) = symbols::tree_sitter_language(source.detection.language) else {
        return Vec::new();
    };

    if parser.set_language(&language).is_err() {
        return Vec::new();
    }
    let Some(tree) = parser.parse(&source.text, None) else {
        return Vec::new();
    };

    let mut ranges = Vec::new();
    collect_ignored_ranges(tree.root_node(), &mut ranges);
    ranges
}

fn collect_ignored_ranges(node: Node<'_>, ranges: &mut Vec<(usize, usize)>) {
    if node.kind() == "comment"
        || node.kind().ends_with("string_literal")
        || node.kind() == "char_literal"
    {
        ranges.push((node.start_byte(), node.end_byte()));
        return;
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_ignored_ranges(child, ranges);
    }
}
