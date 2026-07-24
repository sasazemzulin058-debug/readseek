// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

use std::rc::Rc;

use crate::engine::hash::LineHash;
use crate::engine::lang::{AnalysisEngine, DocumentKind, Language};
use crate::engine::output::{Diagnostic, DiagnosticKind};
use crate::engine::paths::{bytes_contain_identifier, identifier_spans};
use crate::engine::source::{SourceFile, SourceLine, SourceMap, Symbol};
use anyhow::{Result, anyhow};
use tree_sitter::{Node, Parser};

pub(crate) fn parse_source_map(source: &SourceFile) -> Result<SourceMap> {
    let symbols = Vec::new();

    if source.kind != DocumentKind::Source {
        return Ok(SourceMap { symbols });
    }

    match source.detection.engine {
        Some(AnalysisEngine::TreeSitter) => parse_tree_sitter_source_map(source),
        _ => Ok(SourceMap { symbols }),
    }
}

fn parse_tree_sitter_source_map(source: &SourceFile) -> Result<SourceMap> {
    let mut symbols = Vec::new();

    let Some(language) = tree_sitter_language(source.detection.language) else {
        return Ok(SourceMap { symbols });
    };

    if !source.detection.language.has_symbols() {
        return Ok(SourceMap { symbols });
    }

    let mut parser = Parser::new();
    parser
        .set_language(&language)
        .map_err(|error| anyhow!("set tree-sitter language: {error}"))?;
    let tree = parser
        .parse(&source.text, None)
        .ok_or_else(|| anyhow!("tree-sitter parse failed"))?;
    collect_symbols(
        tree.root_node(),
        &source.text,
        source.detection.language,
        None,
        &source.lines,
        &mut symbols,
    );
    symbols.sort_by_key(|symbol| (symbol.start_line, symbol.end_line));

    Ok(SourceMap { symbols })
}

/// Report tree-sitter ERROR and MISSING nodes for a source file.
///
/// Returns an empty list for documents without a tree-sitter engine, so callers
/// can treat "no parser" and "no diagnostics" alike.
pub(crate) fn parse_diagnostics(source: &SourceFile) -> Result<Vec<Diagnostic>> {
    if source.kind != DocumentKind::Source {
        return Ok(Vec::new());
    }
    if source.detection.engine != Some(AnalysisEngine::TreeSitter) {
        return Ok(Vec::new());
    }
    let Some(language) = tree_sitter_language(source.detection.language) else {
        return Ok(Vec::new());
    };

    let mut parser = Parser::new();
    parser
        .set_language(&language)
        .map_err(|error| anyhow!("set tree-sitter language: {error}"))?;
    let tree = parser
        .parse(&source.text, None)
        .ok_or_else(|| anyhow!("tree-sitter parse failed"))?;

    let mut diagnostics = Vec::new();
    collect_diagnostics(tree.root_node(), &mut diagnostics);
    diagnostics.sort_by_key(|diagnostic| (diagnostic.start_line, diagnostic.end_line));
    Ok(diagnostics)
}

fn collect_diagnostics(root: Node<'_>, diagnostics: &mut Vec<Diagnostic>) {
    let mut stack: Vec<Node<'_>> = vec![root];
    while let Some(node) = stack.pop() {
        let kind = if node.is_missing() {
            Some(DiagnosticKind::Missing)
        } else if node.is_error() {
            Some(DiagnosticKind::Error)
        } else {
            None
        };
        if let Some(kind) = kind {
            let (start_line, end_line) = node_line_range(node);
            diagnostics.push(Diagnostic {
                kind,
                start_line,
                end_line,
            });
        }

        // Push in reverse to preserve DFS order.
        for index in (0..node.child_count()).rev() {
            if let Some(child) = node.child(index) {
                stack.push(child);
            }
        }
    }
}

pub(crate) fn tree_sitter_language(language: Language) -> Option<tree_sitter::Language> {
    let language = match language {
        Language::Assembly => tree_sitter_asm::LANGUAGE.into(),
        Language::Bash => tree_sitter_bash::LANGUAGE.into(),
        Language::C => tree_sitter_c::LANGUAGE.into(),
        Language::Cpp => tree_sitter_cpp::LANGUAGE.into(),
        Language::CSharp => tree_sitter_c_sharp::LANGUAGE.into(),
        Language::Css => tree_sitter_css::LANGUAGE.into(),
        Language::Dockerfile => tree_sitter_containerfile::LANGUAGE.into(),
        Language::Go => tree_sitter_go::LANGUAGE.into(),
        Language::Gdscript => tree_sitter_gdscript::LANGUAGE.into(),
        Language::Java => tree_sitter_java::LANGUAGE.into(),
        Language::JavaScript | Language::Jsx => tree_sitter_javascript::LANGUAGE.into(),
        Language::Html => tree_sitter_html::LANGUAGE.into(),
        Language::Json => tree_sitter_json::LANGUAGE.into(),
        Language::Xml => tree_sitter_xml::LANGUAGE_XML.into(),
        Language::Yaml => tree_sitter_yaml::LANGUAGE.into(),
        Language::Just => tree_sitter_just::LANGUAGE.into(),
        Language::Kconfig => tree_sitter_kconfig::LANGUAGE.into(),
        Language::Latex => codebook_tree_sitter_latex::LANGUAGE.into(),
        Language::Lua => tree_sitter_lua::LANGUAGE.into(),
        Language::Make => tree_sitter_make::LANGUAGE.into(),
        Language::Markdown => tree_sitter_md_025::LANGUAGE.into(),
        Language::Meson => arborium_meson::language().into(),
        Language::Nix => tree_sitter_nix::LANGUAGE.into(),
        Language::Perl => ts_parser_perl::LANGUAGE.into(),
        Language::Python => tree_sitter_python::LANGUAGE.into(),
        Language::Php => tree_sitter_php::LANGUAGE_PHP.into(),
        Language::Puppet => tree_sitter_puppet::LANGUAGE.into(),
        Language::Ruby => tree_sitter_ruby::LANGUAGE.into(),
        Language::Riscv => tree_sitter_riscv::LANGUAGE.into(),
        Language::Rust => tree_sitter_rust::LANGUAGE.into(),
        Language::Swift => tree_sitter_swift::LANGUAGE.into(),
        Language::Sql => tree_sitter_sequel::LANGUAGE.into(),
        Language::Typst => codebook_tree_sitter_typst::LANGUAGE.into(),
        Language::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        Language::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
        Language::Toml => tree_sitter_toml_ng::LANGUAGE.into(),
        Language::Vimscript => tree_sitter_vim::language(),
        Language::Zig => tree_sitter_zig::LANGUAGE.into(),
        Language::Unknown => return None,
    };

    Some(language)
}

fn collect_symbols(
    root: Node<'_>,
    source: &str,
    language: Language,
    parent: Option<&str>,
    lines: &[SourceLine],
    symbols: &mut Vec<Symbol>,
) {
    // Iterative DFS using an explicit stack to avoid stack overflow on
    // deep ASTs (the tree-sitter CST for a single C file can be thousands
    // of nodes deep in debug builds where each stack frame is large).
    let mut stack: Vec<(Node<'_>, Option<Rc<str>>)> = vec![(root, parent.map(Rc::from))];
    while let Some((node, current_parent)) = stack.pop() {
        let symbol = symbol_for_node(node, source, language, current_parent.as_deref(), lines);
        let next_parent = symbol
            .as_ref()
            .map(|symbol| Rc::from(symbol.qualified_name.as_str()))
            .or(current_parent);
        if let Some(symbol) = symbol {
            symbols.push(symbol);
        }

        // Push in reverse to preserve DFS order.
        for index in (0..node.child_count()).rev() {
            if let Some(child) = node.child(index) {
                stack.push((child, next_parent.clone()));
            }
        }
    }
}

fn symbol_for_node(
    node: Node<'_>,
    source: &str,
    language: Language,
    parent: Option<&str>,
    lines: &[SourceLine],
) -> Option<Symbol> {
    let (kind, name) = match language {
        Language::Rust => rust_symbol(node, source),
        Language::TypeScript | Language::Tsx | Language::JavaScript | Language::Jsx => {
            js_like_symbol(node, source)
        }
        Language::Python => python_symbol(node, source),
        Language::Bash => bash_symbol(node, source),
        Language::C | Language::Cpp => c_symbol(node, source),
        Language::CSharp => csharp_symbol(node, source),
        Language::Go => go_symbol(node, source),
        Language::Java => java_symbol(node, source),
        Language::Just => just_symbol(node, source),
        Language::Kconfig => kconfig_symbol(node, source),
        Language::Markdown => markdown_symbol(node, source),
        Language::Make => make_symbol(node, source),
        Language::Php => php_symbol(node, source),
        Language::Ruby => ruby_symbol(node, source),
        Language::Swift => swift_symbol(node, source),
        Language::Vimscript => vimscript_symbol(node, source),
        _ => return None,
    }?;

    let (start_line, end_line) = if language == Language::Markdown && node.kind() == "atx_heading" {
        let start = node.start_position().row + 1;
        (start, start)
    } else {
        node_line_range(node)
    };
    let start_hash = line_hash(lines, start_line)?;
    let end_hash = line_hash(lines, end_line)?;
    let qualified_name = parent.map_or_else(|| name.clone(), |parent| format!("{parent}.{name}"));
    let name_byte = name_byte_in(node, source, &name);

    Some(Symbol {
        kind,
        name,
        qualified_name,
        start_line,
        end_line,
        start_hash,
        end_hash,
        start_byte: node.start_byte(),
        end_byte: node.end_byte(),
        name_byte,
    })
}

/// Byte offset of the symbol's name token within its node span.
///
/// Locates `name` as an identifier-bounded token inside the node, falling back
/// to the node start when the name is synthetic (e.g. a Swift signature or a
/// Markdown heading) and so has no single matching token.
fn name_byte_in(node: Node<'_>, source: &str, name: &str) -> usize {
    let start = node.start_byte();
    let span = source.get(start..node.end_byte()).unwrap_or("");
    identifier_spans(span.as_bytes(), name.as_bytes())
        .next()
        .map_or(start, |offset| start + offset)
}

/// A token resolved at a byte position via tree-sitter.
pub(crate) struct Token {
    pub(crate) text: String,
    pub(crate) start_byte: usize,
    pub(crate) end_byte: usize,
}

/// Resolve the identifier-like leaf token covering `byte` using tree-sitter.
///
/// Returns `None` when the language has no parser, the parse fails, or the
/// covering leaf is not an identifier. Callers fall back to a byte scan.
pub(crate) fn token_at(source: &SourceFile, byte: usize) -> Option<Token> {
    if source.detection.engine != Some(AnalysisEngine::TreeSitter) {
        return None;
    }
    let language = tree_sitter_language(source.detection.language)?;
    let mut parser = Parser::new();
    parser.set_language(&language).ok()?;
    let tree = parser.parse(&source.text, None)?;

    let lookup = byte.min(source.text.len().saturating_sub(1));
    let node = named_leaf_at(tree.root_node(), lookup)?;
    if !is_identifier_node(node.kind()) {
        return None;
    }
    let text = node.utf8_text(source.text.as_bytes()).ok()?;
    Some(Token {
        text: text.to_owned(),
        start_byte: node.start_byte(),
        end_byte: node.end_byte(),
    })
}

fn named_leaf_at(node: Node<'_>, byte: usize) -> Option<Node<'_>> {
    let mut current = node.descendant_for_byte_range(byte, byte)?;
    while current.named_child_count() > 0 {
        match current.named_descendant_for_byte_range(byte, byte) {
            Some(child) if child.id() != current.id() => current = child,
            _ => break,
        }
    }
    Some(current)
}

fn is_identifier_node(kind: &str) -> bool {
    kind == "identifier" || kind.ends_with("_identifier") || matches!(kind, "word" | "name")
}

pub(crate) fn node_line_range(node: Node<'_>) -> (usize, usize) {
    let start_line = node.start_position().row + 1;
    let end_position = node.end_position();
    let end_line = if end_position.column == 0 && end_position.row + 1 > start_line {
        end_position.row
    } else {
        end_position.row + 1
    };

    (start_line, end_line)
}

fn rust_symbol(node: Node<'_>, source: &str) -> Option<(String, String)> {
    match node.kind() {
        "function_item" => named_symbol(node, source, "name", "function"),
        "struct_item" => named_symbol(node, source, "name", "struct"),
        "enum_item" => named_symbol(node, source, "name", "enum"),
        "trait_item" => named_symbol(node, source, "name", "trait"),
        "impl_item" => named_symbol(node, source, "type", "impl"),
        "mod_item" => named_symbol(node, source, "name", "module"),
        _ => None,
    }
}

fn js_like_symbol(node: Node<'_>, source: &str) -> Option<(String, String)> {
    match node.kind() {
        "function_declaration" | "generator_function_declaration" => {
            named_symbol(node, source, "name", "function")
        }
        "method_definition" => named_symbol(node, source, "name", "method"),
        "class_declaration" => named_symbol(node, source, "name", "class"),
        "interface_declaration" => named_symbol(node, source, "name", "interface"),
        "type_alias_declaration" => named_symbol(node, source, "name", "type"),
        "lexical_declaration" | "variable_declaration" => {
            let mut cursor = node.walk();
            let mut result = None;
            for child in node.children(&mut cursor) {
                if child.kind() != "variable_declarator" {
                    continue;
                }
                let value = child.child_by_field_name("value")?;
                if !matches!(value.kind(), "arrow_function" | "function_expression") {
                    continue;
                }
                result =
                    named_child(child, source, "name").map(|name| ("function".to_owned(), name));
                break;
            }
            result
        }
        _ => None,
    }
}

fn python_symbol(node: Node<'_>, source: &str) -> Option<(String, String)> {
    match node.kind() {
        "function_definition" => named_symbol(node, source, "name", "function"),
        "class_definition" => named_symbol(node, source, "name", "class"),
        _ => None,
    }
}

fn just_symbol(node: Node<'_>, source: &str) -> Option<(String, String)> {
    match node.kind() {
        "recipe_header" => named_symbol(node, source, "name", "recipe"),
        _ => None,
    }
}

fn kconfig_symbol(node: Node<'_>, source: &str) -> Option<(String, String)> {
    match node.kind() {
        "config" | "menuconfig" => named_symbol(node, source, "name", node.kind()),
        "menu" => child_text(node, source, "name")
            .map(|name| ("menu".to_owned(), name.trim_matches('"').to_owned())),
        _ => None,
    }
}

fn make_symbol(node: Node<'_>, source: &str) -> Option<(String, String)> {
    match node.kind() {
        "rule" => descendant_identifier(node, source).map(|name| ("target".to_owned(), name)),
        _ => None,
    }
}

fn markdown_symbol(node: Node<'_>, source: &str) -> Option<(String, String)> {
    match node.kind() {
        "atx_heading" | "setext_heading" => {
            let text = node.utf8_text(source.as_bytes()).ok()?.trim();
            let name = text
                .trim_start_matches('#')
                .trim()
                .trim_end_matches('#')
                .trim();
            name.lines()
                .next()
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(|name| ("heading".to_owned(), name.to_owned()))
        }
        _ => None,
    }
}

fn bash_symbol(node: Node<'_>, source: &str) -> Option<(String, String)> {
    match node.kind() {
        "function_definition" => {
            descendant_identifier(node, source).map(|name| ("function".to_owned(), name))
        }
        _ => None,
    }
}

fn vimscript_symbol(node: Node<'_>, source: &str) -> Option<(String, String)> {
    match node.kind() {
        "function_definition" => {
            let mut cursor = node.walk();
            node.named_children(&mut cursor)
                .find(|child| child.kind() == "function_declaration")
                .and_then(|declaration| named_child(declaration, source, "name"))
                .map(|name| ("function".to_owned(), name))
        }
        _ => None,
    }
}

fn c_symbol(node: Node<'_>, source: &str) -> Option<(String, String)> {
    match node.kind() {
        "function_definition" => node
            .child_by_field_name("declarator")
            .and_then(|declarator| declarator_name(declarator, source))
            .or_else(|| descendant_identifier(node, source))
            .map(|name| ("function".to_owned(), name)),
        "field_declaration" => node
            .child_by_field_name("declarator")
            .and_then(|declarator| declarator_name(declarator, source))
            .map(|name| ("member".to_owned(), name)),
        "struct_specifier" => named_symbol(node, source, "name", "struct"),
        "enumerator" => named_symbol(node, source, "name", "enumerator"),
        "enum_specifier" => named_symbol(node, source, "name", "enum"),
        "class_specifier" => named_symbol(node, source, "name", "class"),
        "namespace_definition" | "namespace_alias_definition" => {
            named_symbol(node, source, "name", "namespace")
        }
        "declaration" | "type_definition" => {
            let text = node.utf8_text(source.as_bytes()).ok()?;
            if bytes_contain_identifier(text.as_bytes(), b"typedef") {
                node.child_by_field_name("declarator")
                    .and_then(|declarator| declarator_identifier(declarator, source))
                    .or_else(|| last_identifier(text))
                    .map(|name| ("type".to_owned(), name))
            } else if node.kind() == "declaration" && {
                let mut current = node;
                loop {
                    match current.parent() {
                        Some(parent) => match parent.kind() {
                            "translation_unit"
                            | "namespace_definition"
                            | "linkage_specification" => {
                                break true;
                            }
                            "function_definition"
                            | "compound_statement"
                            | "class_specifier"
                            | "struct_specifier"
                            | "enum_specifier" => {
                                break false;
                            }
                            _ => current = parent,
                        },
                        None => break false,
                    }
                }
            } {
                if let Some(function) = descendant_of_kind(node, "function_declarator") {
                    function
                        .child_by_field_name("declarator")
                        .and_then(|declarator| declarator_identifier(declarator, source))
                        .map(|name| ("function".to_owned(), name))
                } else {
                    node.child_by_field_name("declarator")
                        .and_then(|declarator| declarator_name(declarator, source))
                        .map(|name| ("variable".to_owned(), name))
                }
            } else {
                None
            }
        }
        "preproc_def" | "preproc_function_def" => {
            descendant_identifier(node, source).map(|name| ("macro".to_owned(), name))
        }
        "alias_declaration" => named_symbol(node, source, "name", "type"),
        "concept_definition" => named_symbol(node, source, "name", "concept"),
        _ => None,
    }
}

fn descendant_of_kind<'tree>(root: Node<'tree>, kind: &str) -> Option<Node<'tree>> {
    let mut stack: Vec<Node<'tree>> = vec![root];
    while let Some(node) = stack.pop() {
        if node.kind() == kind {
            return Some(node);
        }

        for index in (0..node.named_child_count()).rev() {
            if let Some(child) = node.named_child(index) {
                stack.push(child);
            }
        }
    }

    None
}

fn declarator_name(node: Node<'_>, source: &str) -> Option<String> {
    match node.kind() {
        "identifier"
        | "field_identifier"
        | "destructor_name"
        | "operator_name"
        | "qualified_identifier" => node
            .utf8_text(source.as_bytes())
            .ok()
            .map(ToOwned::to_owned),
        _ => declarator_name(node.child_by_field_name("declarator")?, source),
    }
}

fn declarator_identifier(node: Node<'_>, source: &str) -> Option<String> {
    if matches!(
        node.kind(),
        "identifier" | "type_identifier" | "field_identifier"
    ) {
        return node
            .utf8_text(source.as_bytes())
            .ok()
            .map(ToOwned::to_owned);
    }

    for index in (0..node.named_child_count()).rev() {
        if let Some(child) = node.named_child(index)
            && let Some(name) = declarator_identifier(child, source)
        {
            return Some(name);
        }
    }

    None
}

fn last_identifier(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut index = bytes.len();
    while index > 0 {
        index -= 1;
        if !(bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_') {
            continue;
        }

        let end = index + 1;
        while index > 0 && (bytes[index - 1].is_ascii_alphanumeric() || bytes[index - 1] == b'_') {
            index -= 1;
        }
        if bytes[index].is_ascii_digit() {
            continue;
        }
        return text.get(index..end).map(ToOwned::to_owned);
    }

    None
}

fn csharp_symbol(node: Node<'_>, source: &str) -> Option<(String, String)> {
    match node.kind() {
        "method_declaration" => named_symbol(node, source, "name", "method"),
        "constructor_declaration" => named_symbol(node, source, "name", "constructor"),
        "class_declaration" => named_symbol(node, source, "name", "class"),
        "interface_declaration" => named_symbol(node, source, "name", "interface"),
        "struct_declaration" => named_symbol(node, source, "name", "struct"),
        "enum_declaration" => named_symbol(node, source, "name", "enum"),
        "namespace_declaration" => named_symbol(node, source, "name", "namespace"),
        "delegate_declaration" => named_symbol(node, source, "name", "delegate"),
        "record_declaration" => named_symbol(node, source, "name", "record"),
        _ => None,
    }
}

fn go_symbol(node: Node<'_>, source: &str) -> Option<(String, String)> {
    match node.kind() {
        "function_declaration" => named_symbol(node, source, "name", "function"),
        "method_declaration" => named_symbol(node, source, "name", "method"),
        "type_spec" | "type_alias" => named_symbol(node, source, "name", "type"),
        _ => None,
    }
}

fn java_symbol(node: Node<'_>, source: &str) -> Option<(String, String)> {
    match node.kind() {
        "method_declaration" => named_symbol(node, source, "name", "method"),
        "constructor_declaration" => named_symbol(node, source, "name", "constructor"),
        "class_declaration" => named_symbol(node, source, "name", "class"),
        "interface_declaration" => named_symbol(node, source, "name", "interface"),
        "enum_declaration" => named_symbol(node, source, "name", "enum"),
        "record_declaration" => named_symbol(node, source, "name", "record"),
        "annotation_type_declaration" => named_symbol(node, source, "name", "annotation"),
        "module_declaration" => named_symbol(node, source, "name", "module"),
        _ => None,
    }
}

fn php_symbol(node: Node<'_>, source: &str) -> Option<(String, String)> {
    match node.kind() {
        "function_definition" => named_symbol(node, source, "name", "function"),
        "method_declaration" => named_symbol(node, source, "name", "method"),
        "class_declaration" => named_symbol(node, source, "name", "class"),
        "interface_declaration" => named_symbol(node, source, "name", "interface"),
        "trait_declaration" => named_symbol(node, source, "name", "trait"),
        "enum_declaration" => named_symbol(node, source, "name", "enum"),
        "namespace_definition" => named_symbol(node, source, "name", "namespace"),
        _ => None,
    }
}

fn ruby_symbol(node: Node<'_>, source: &str) -> Option<(String, String)> {
    match node.kind() {
        "method" | "singleton_method" => named_symbol(node, source, "name", "method"),
        "class" => named_symbol(node, source, "name", "class"),
        "module" => named_symbol(node, source, "name", "module"),
        _ => None,
    }
}

fn swift_symbol(node: Node<'_>, source: &str) -> Option<(String, String)> {
    match node.kind() {
        "class_declaration" => {
            let declaration_kind = child_text(node, source, "declaration_kind")?;
            let name = named_child(node, source, "name")?;
            Some(("class".to_owned(), format!("{declaration_kind} {name}")))
        }
        "function_declaration" | "protocol_function_declaration" => {
            let name = child_text(node, source, "name")?;
            let start_byte = node.start_byte();
            let end_byte = node
                .child_by_field_name("body")
                .map_or_else(|| node.end_byte(), |body| body.start_byte());
            let prefix = source.get(start_byte..end_byte)?.trim();
            let func_name = if prefix.starts_with("func ")
                || prefix.starts_with("static func ")
                || prefix.starts_with("class func ")
            {
                prefix.trim_end().to_owned()
            } else {
                name
            };
            Some(("function".to_owned(), func_name))
        }
        "deinit_declaration" => Some(("deinit".to_owned(), "deinit".to_owned())),
        _ => None,
    }
}

fn child_text(node: Node<'_>, source: &str, field: &str) -> Option<String> {
    node.child_by_field_name(field)
        .and_then(|child| child.utf8_text(source.as_bytes()).ok())
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned)
}

fn named_child(node: Node<'_>, source: &str, field: &str) -> Option<String> {
    node.child_by_field_name(field)
        .and_then(|child| child.utf8_text(source.as_bytes()).ok())
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
}

fn named_symbol(node: Node<'_>, source: &str, field: &str, kind: &str) -> Option<(String, String)> {
    named_child(node, source, field).map(|name| (kind.to_owned(), name))
}

fn descendant_identifier(node: Node<'_>, source: &str) -> Option<String> {
    if matches!(
        node.kind(),
        "identifier" | "type_identifier" | "word" | "variable_name"
    ) {
        return node
            .utf8_text(source.as_bytes())
            .ok()
            .map(ToOwned::to_owned);
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if let Some(name) = descendant_identifier(child, source) {
            return Some(name);
        }
    }

    None
}

fn line_hash(lines: &[SourceLine], line: usize) -> Option<LineHash> {
    lines.get(line.checked_sub(1)?).map(SourceLine::hash)
}
