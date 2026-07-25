// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

//! Per-file scope and binding resolution.
//!
//! Given a cursor position, this resolves which other occurrences in the same
//! file bind to the *same* declaration, so callers can distinguish a local from
//! a same-named binding in another scope. Resolution is conservative: it is only
//! attempted for languages with a binding table below, and only for names that
//! resolve to a lexical declaration. Everything else is reported as unresolved
//! so callers can fall back to name matching without silently over-matching.

use crate::engine::lang::Language;
use crate::engine::source::SourceFile;
use crate::engine::symbols::tree_sitter_language;
use serde::Serialize;
use tree_sitter::{Node, Parser, Tree};

mod cpp;
mod csharp;
mod go;
mod java;
mod python;
mod rust;
mod swift;
mod typescript;
mod vimscript;

/// How an occurrence relates to the resolved binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OccurrenceKind {
    /// The identifier that introduces the binding.
    Definition,
    /// A use that resolves to the binding.
    Reference,
    /// Same name, but resolves to a different binding (shadowed or unrelated).
    Shadowed,
}

/// One resolved occurrence of the target name within a file.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Occurrence {
    pub(crate) start_byte: usize,
    pub(crate) end_byte: usize,
    pub(crate) kind: OccurrenceKind,
}

/// The binding the cursor token resolves to, with every same-file occurrence
/// classified relative to it.
#[derive(Debug)]
pub(crate) struct Binding {
    pub(crate) name: String,
    pub(crate) occurrences: Vec<Occurrence>,
}

/// A name collision a rename to `new_name` would introduce.
#[derive(Debug)]
pub(crate) struct Conflict {
    pub(crate) byte: usize,
    pub(crate) reason: String,
}

/// Parse `source` and return its binding table and syntax tree, or `None` when
/// the language has no binding support or the parse fails.
fn parse_source(source: &SourceFile) -> Option<(&'static dyn LanguageBindingRule, Tree)> {
    let table = binding_rules(source.detection.language)?;
    let language = tree_sitter_language(source.detection.language)?;
    let mut parser = Parser::new();
    parser.set_language(&language).ok()?;
    let tree = parser.parse(&source.text, None)?;
    Some((table, tree))
}

/// Resolve the lexical binding for the identifier covering `byte`.
///
/// Returns `None` when the language is unsupported, the parse fails, the cursor
/// is not on an identifier, or the name has no resolvable lexical declaration.
pub(crate) fn resolve(source: &SourceFile, byte: usize) -> Option<Binding> {
    resolve_with_conflicts(source, byte, None).map(|(binding, _)| binding)
}

/// Resolve the binding and, when `new_name` is given, report rename conflicts.
///
/// A conflict is reported when `new_name` already resolves to a declaration that
/// is visible from a renamed occurrence; renaming would then change which
/// declaration that occurrence binds to. This is conservative: it flags possible
/// capture rather than proving it.
pub(crate) fn resolve_with_conflicts(
    source: &SourceFile,
    byte: usize,
    new_name: Option<&str>,
) -> Option<(Binding, Vec<Conflict>)> {
    let (table, tree) = parse_source(source)?;
    let root = tree.root_node();
    let src = source.text.as_bytes();

    let lookup = byte.min(source.text.len().saturating_sub(1));
    let cursor = identifier_leaf(root.descendant_for_byte_range(lookup, lookup)?, table)?;
    let name = cursor.utf8_text(src).ok()?.to_owned();

    let mut declarations = Vec::new();
    collect_declarations(root, src, table, &mut declarations);

    let target_def = resolve_node(cursor, &name, &declarations, table)?;

    let mut occurrences = Vec::new();
    collect_occurrences(
        root,
        src,
        table,
        &name,
        target_def,
        &declarations,
        &mut occurrences,
    );
    occurrences.sort_by_key(|occurrence| occurrence.start_byte);

    let conflicts = new_name
        .map(|new_name| find_conflicts(root, new_name, &occurrences, &declarations, table))
        .unwrap_or_default();

    Some((Binding { name, occurrences }, conflicts))
}

/// The identifier text under `byte`, if the cursor sits on one.
///
/// Unlike [`resolve`], this does not require the name to bind to a local
/// declaration, so it also names top-level symbols (functions, types) the
/// resolver does not track but a cross-file rename still targets.
pub(crate) fn identifier_at(source: &SourceFile, byte: usize) -> Option<String> {
    if let Some((table, tree)) = parse_source(source) {
        let root = tree.root_node();
        let lookup = byte.min(source.text.len().saturating_sub(1));
        if let Some(leaf) = root
            .descendant_for_byte_range(lookup, lookup)
            .and_then(|node| identifier_leaf(node, table))
        {
            return leaf
                .utf8_text(source.text.as_bytes())
                .ok()
                .map(str::to_owned);
        }
    }
    // Byte-level fallback for identifiers inside preprocessor bodies
    // (tree-sitter-c treats #define bodies as opaque `preproc_arg` nodes),
    // parse-error subtrees, and languages without a binding table.
    identifier_at_byte(source.text.as_bytes(), byte)
}

/// Extract the identifier covering `byte` with a byte-level scan.
///
/// Scans backward and forward for identifier characters; returns `None` when
/// the cursor byte is not an identifier character or the extracted span starts
/// with a digit. This is the fallback when tree-sitter cannot resolve the token
/// (e.g. inside `#define` bodies, which tree-sitter-c stores as opaque
/// `preproc_arg` nodes).
fn identifier_at_byte(text: &[u8], byte: usize) -> Option<String> {
    let pos = byte.min(text.len().saturating_sub(1));
    if !text[pos].is_ascii_alphanumeric() && text[pos] != b'_' {
        return None;
    }
    let mut start = pos;
    while start > 0 && (text[start - 1].is_ascii_alphanumeric() || text[start - 1] == b'_') {
        start -= 1;
    }
    // Reject spans starting with a digit: they are number literals, not
    // identifiers, even inside preprocessor bodies.
    if text[start].is_ascii_digit() {
        return None;
    }
    let mut end = pos + 1;
    while end < text.len() && (text[end].is_ascii_alphanumeric() || text[end] == b'_') {
        end += 1;
    }
    String::from_utf8(text[start..end].to_vec()).ok()
}

/// Name occurrences a cross-file rename should touch within one file.
pub(crate) struct CrossFileMatches {
    /// Byte spans of free uses of the old name to rename.
    pub(crate) occurrences: Vec<(usize, usize)>,
    /// Byte offsets of same-named local declarations and uses that resolve to them.
    pub(crate) excluded: Vec<usize>,
    /// Byte offsets where renaming to the new name would capture a local binding.
    pub(crate) conflicts: Vec<usize>,
}

/// Find the occurrences of `name` in a file that take part in a cross-file
/// rename to `new_name`.
///
/// readseek has no cross-file symbol resolver, so a use in another file is taken
/// to reference the cross-file target only when it does *not* resolve to a local
/// declaration here; an occurrence that binds to a local declaration is a shadow
/// and is left untouched. Returns `None` for languages without a binding table,
/// so the caller can fall back to a plain name scan.
pub(crate) fn cross_file_matches(
    source: &SourceFile,
    name: &str,
    new_name: &str,
) -> Option<CrossFileMatches> {
    let (table, tree) = parse_source(source)?;
    let root = tree.root_node();
    let src = source.text.as_bytes();

    let mut declarations = Vec::new();
    collect_declarations(root, src, table, &mut declarations);

    let mut matches = CrossFileMatches {
        occurrences: Vec::new(),
        excluded: declarations
            .iter()
            .filter(|declaration| declaration.name == name)
            .map(|declaration| declaration.ident.start_byte())
            .collect(),
        conflicts: Vec::new(),
    };
    collect_free_occurrences(
        root,
        src,
        table,
        name,
        new_name,
        &declarations,
        &mut matches,
    );
    matches.occurrences.sort_by_key(|&(start, _)| start);
    matches.excluded.sort_unstable();
    matches.excluded.dedup();
    matches.conflicts.sort_unstable();
    Some(matches)
}

/// Walk the tree collecting uses of `name` that do not resolve to a local
/// declaration, marking same-named local declarations and shadowed uses so a
/// caller can exclude them from a byte-level name scan.
fn collect_free_occurrences(
    root: Node<'_>,
    src: &[u8],
    table: &dyn LanguageBindingRule,
    name: &str,
    new_name: &str,
    declarations: &[Declaration<'_>],
    out: &mut CrossFileMatches,
) {
    let mut stack: Vec<Node<'_>> = vec![root];
    while let Some(node) = stack.pop() {
        if is_identifier_kind(node.kind(), table)
            && node.child_count() == 0
            && table.is_reference(node)
            && node.utf8_text(src) == Ok(name)
        {
            let resolves_old = resolve_node(node, name, declarations, table);
            if resolves_old.is_none() {
                out.occurrences.push((node.start_byte(), node.end_byte()));
                if resolve_node(node, new_name, declarations, table).is_some() {
                    out.conflicts.push(node.start_byte());
                }
            } else {
                out.excluded.push(node.start_byte());
            }
        }
        for index in (0..node.child_count()).rev() {
            if let Some(child) = node.child(index) {
                stack.push(child);
            }
        }
    }
}

/// Check whether renaming the binding to `new_name` would capture or be captured.
fn find_conflicts(
    root: Node<'_>,
    new_name: &str,
    occurrences: &[Occurrence],
    declarations: &[Declaration<'_>],
    table: &dyn LanguageBindingRule,
) -> Vec<Conflict> {
    occurrences
        .iter()
        .filter_map(|occurrence| {
            let byte = occurrence.start_byte;
            let node = root.descendant_for_byte_range(byte, byte)?;
            resolve_node(node, new_name, declarations, table)
                .is_some()
                .then(|| Conflict {
                    byte,
                    reason: format!("`{new_name}` already resolves to a binding here"),
                })
        })
        .collect()
}

/// A declared name together with the identifier node and the scope it lives in.
struct Declaration<'tree> {
    name: String,
    ident: Node<'tree>,
    scope: usize,
}

/// The innermost enclosing scope node's id, or the root id when none applies.
///
/// An identifier that escapes its scope (a parameter default or leading
/// comprehension iterable) is placed in the scope enclosing its nearest
/// syntactic scope, matching where Python evaluates it.
fn scope_of(node: Node<'_>, table: &dyn LanguageBindingRule) -> usize {
    let mut current = if table.escapes_scope(node) {
        enclosing_scope(node, table).and_then(|scope| scope.parent())
    } else {
        node.parent()
    };
    while let Some(parent) = current {
        if table.scope_kinds().contains(&parent.kind()) && !table.binds_past(node, parent.kind()) {
            return parent.id();
        }
        current = parent.parent();
    }
    node_root(node).id()
}

/// The nearest ancestor of `node` that opens a scope, if any.
fn enclosing_scope<'tree>(
    node: Node<'tree>,
    table: &dyn LanguageBindingRule,
) -> Option<Node<'tree>> {
    let mut current = node.parent();
    while let Some(parent) = current {
        if table.scope_kinds().contains(&parent.kind()) {
            return Some(parent);
        }
        current = parent.parent();
    }
    None
}

fn node_root(node: Node<'_>) -> Node<'_> {
    let mut current = node;
    while let Some(parent) = current.parent() {
        current = parent;
    }
    current
}

fn collect_declarations<'tree>(
    root: Node<'tree>,
    src: &[u8],
    table: &dyn LanguageBindingRule,
    out: &mut Vec<Declaration<'tree>>,
) {
    let mut stack: Vec<Node<'tree>> = vec![root];
    while let Some(node) = stack.pop() {
        for ident in table.declared_idents(node, src) {
            if let Ok(name) = ident.utf8_text(src) {
                out.push(Declaration {
                    name: name.to_owned(),
                    ident,
                    scope: scope_of(ident, table),
                });
            }
        }
        for index in (0..node.child_count()).rev() {
            if let Some(child) = node.child(index) {
                stack.push(child);
            }
        }
    }
}

/// Resolve an identifier node to the byte offset of its binding declaration.
///
/// Walks enclosing scopes innermost-first and, within the nearest scope that
/// declares the name, picks the binding according to `table.resolution`: under
/// [`Resolution::Lexical`] the nearest lexically-preceding declaration (so a
/// re-declaration shadows), under [`Resolution::Hoisted`] the first declaration
/// in the scope (so all same-name declarations are one binding). Class scopes
/// are consulted only for uses in their own direct body, and an escaping
/// identifier resolves from the scope enclosing its nearest syntactic scope.
fn resolve_node(
    node: Node<'_>,
    name: &str,
    declarations: &[Declaration<'_>],
    table: &dyn LanguageBindingRule,
) -> Option<usize> {
    let use_start = node.start_byte();
    let start = if table.escapes_scope(node) {
        enclosing_scope(node, table).and_then(|scope| scope.parent())?
    } else {
        node
    };
    let mut scope = Some(start);
    let mut left_innermost_scope = false;
    while let Some(current) = scope {
        let is_scope = current.parent().is_none() || table.scope_kinds().contains(&current.kind());
        let hidden_class =
            left_innermost_scope && table.class_scope_kinds().contains(&current.kind());
        if !hidden_class {
            let scoped = || {
                declarations.iter().filter(|declaration| {
                    declaration.name == name && declaration.scope == current.id()
                })
            };
            let resolved = match table.resolution() {
                Resolution::Lexical
                    if scoped()
                        .any(|declaration| table.unifies_declarations(declaration.ident)) =>
                {
                    scoped().min_by_key(|declaration| declaration.ident.start_byte())
                }
                Resolution::Lexical => scoped()
                    .filter(|declaration| declaration.ident.start_byte() <= use_start)
                    .max_by_key(|declaration| declaration.ident.start_byte()),
                Resolution::Hoisted => {
                    scoped().min_by_key(|declaration| declaration.ident.start_byte())
                }
            };
            if let Some(declaration) = resolved {
                return Some(declaration.ident.start_byte());
            }
        }
        if is_scope {
            left_innermost_scope = true;
        }
        scope = current.parent();
    }
    None
}

fn collect_occurrences(
    root: Node<'_>,
    src: &[u8],
    table: &dyn LanguageBindingRule,
    name: &str,
    target_def: usize,
    declarations: &[Declaration<'_>],
    out: &mut Vec<Occurrence>,
) {
    let mut stack: Vec<Node<'_>> = vec![root];
    while let Some(node) = stack.pop() {
        if is_identifier_kind(node.kind(), table)
            && node.child_count() == 0
            && table.is_reference(node)
            && node.utf8_text(src) == Ok(name)
        {
            let resolved = resolve_node(node, name, declarations, table);
            let kind = if resolved == Some(target_def) {
                if node.start_byte() == target_def {
                    OccurrenceKind::Definition
                } else {
                    OccurrenceKind::Reference
                }
            } else {
                OccurrenceKind::Shadowed
            };
            out.push(Occurrence {
                start_byte: node.start_byte(),
                end_byte: node.end_byte(),
                kind,
            });
        }
        for index in (0..node.child_count()).rev() {
            if let Some(child) = node.child(index) {
                stack.push(child);
            }
        }
    }
}

fn identifier_leaf<'tree>(
    node: Node<'tree>,
    table: &dyn LanguageBindingRule,
) -> Option<Node<'tree>> {
    let mut current = node;
    while current.named_child_count() > 0 {
        let byte = current.start_byte();
        match current.named_descendant_for_byte_range(byte, byte) {
            Some(child) if child.id() != current.id() => current = child,
            _ => break,
        }
    }
    (is_identifier_kind(current.kind(), table) && table.is_reference(current)).then_some(current)
}

fn is_identifier_kind(kind: &str, table: &dyn LanguageBindingRule) -> bool {
    table.identifier_kinds().contains(&kind)
}

/// How repeated declarations of one name within a scope relate to each other.
#[derive(Clone, Copy)]
pub(crate) enum Resolution {
    Lexical,
    Hoisted,
}

/// Per-language description of scopes and the identifiers that introduce bindings.
pub(crate) trait LanguageBindingRule: Sync {
    fn scope_kinds(&self) -> &'static [&'static str];
    fn class_scope_kinds(&self) -> &'static [&'static str] {
        &[]
    }
    fn identifier_kinds(&self) -> &'static [&'static str];
    fn declared_idents<'a>(&self, node: Node<'a>, src: &[u8]) -> Vec<Node<'a>>;
    fn resolution(&self) -> Resolution {
        Resolution::Lexical
    }
    fn is_reference(&self, _node: Node<'_>) -> bool {
        true
    }
    fn escapes_scope(&self, _node: Node<'_>) -> bool {
        false
    }
    fn binds_past(&self, _node: Node<'_>, _scope_kind: &str) -> bool {
        false
    }
    fn unifies_declarations(&self, _node: Node<'_>) -> bool {
        false
    }
}

fn binding_rules(language: Language) -> Option<&'static dyn LanguageBindingRule> {
    match language {
        Language::Rust => Some(&RUST_RULES),
        Language::C | Language::Cpp => Some(&CPP_RULES),
        Language::Python => Some(&PYTHON_RULES),
        Language::Go => Some(&GO_RULES),
        Language::Java => Some(&JAVA_RULES),
        Language::TypeScript | Language::Tsx | Language::JavaScript | Language::Jsx => {
            Some(&TYPESCRIPT_RULES)
        }
        Language::CSharp => Some(&CSHARP_RULES),
        Language::Swift => Some(&SWIFT_RULES),
        Language::Vimscript => Some(&VIMSCRIPT_RULES),
        _ => None,
    }
}

struct RustRules;
impl LanguageBindingRule for RustRules {
    fn scope_kinds(&self) -> &'static [&'static str] {
        &[
            "block",
            "function_item",
            "closure_expression",
            "match_arm",
            "for_expression",
            "while_expression",
            "if_expression",
        ]
    }
    fn identifier_kinds(&self) -> &'static [&'static str] {
        &["identifier"]
    }
    fn declared_idents<'a>(&self, node: Node<'a>, src: &[u8]) -> Vec<Node<'a>> {
        rust::declared_idents(node, src)
    }
}

struct CppRules;
impl LanguageBindingRule for CppRules {
    fn scope_kinds(&self) -> &'static [&'static str] {
        &[
            "compound_statement",
            "function_definition",
            "if_statement",
            "switch_statement",
            "while_statement",
            "for_statement",
            "for_range_loop",
            "catch_clause",
            "lambda_expression",
        ]
    }
    fn identifier_kinds(&self) -> &'static [&'static str] {
        &["identifier"]
    }
    fn declared_idents<'a>(&self, node: Node<'a>, src: &[u8]) -> Vec<Node<'a>> {
        cpp::declared_idents(node, src)
    }
}

struct PythonRules;
impl LanguageBindingRule for PythonRules {
    fn scope_kinds(&self) -> &'static [&'static str] {
        &[
            "function_definition",
            "lambda",
            "class_definition",
            "list_comprehension",
            "set_comprehension",
            "dictionary_comprehension",
            "generator_expression",
        ]
    }
    fn class_scope_kinds(&self) -> &'static [&'static str] {
        &["class_definition"]
    }
    fn identifier_kinds(&self) -> &'static [&'static str] {
        &["identifier"]
    }
    fn declared_idents<'a>(&self, node: Node<'a>, src: &[u8]) -> Vec<Node<'a>> {
        python::declared_idents(node, src)
    }
    fn resolution(&self) -> Resolution {
        Resolution::Hoisted
    }
    fn is_reference(&self, node: Node<'_>) -> bool {
        python::is_reference(node)
    }
    fn escapes_scope(&self, node: Node<'_>) -> bool {
        python::escapes_scope(node)
    }
    fn binds_past(&self, node: Node<'_>, scope_kind: &str) -> bool {
        python::binds_past(node, scope_kind)
    }
}

struct TypeScriptRules;
impl LanguageBindingRule for TypeScriptRules {
    fn scope_kinds(&self) -> &'static [&'static str] {
        &[
            "statement_block",
            "function_declaration",
            "function_expression",
            "generator_function_declaration",
            "arrow_function",
            "method_definition",
            "class_declaration",
            "for_statement",
            "for_in_statement",
            "catch_clause",
        ]
    }
    fn class_scope_kinds(&self) -> &'static [&'static str] {
        &["class_declaration"]
    }
    fn identifier_kinds(&self) -> &'static [&'static str] {
        &[
            "identifier",
            "shorthand_property_identifier",
            "shorthand_property_identifier_pattern",
        ]
    }
    fn declared_idents<'a>(&self, node: Node<'a>, src: &[u8]) -> Vec<Node<'a>> {
        typescript::declared_idents(node, src)
    }
    fn is_reference(&self, node: Node<'_>) -> bool {
        typescript::is_reference(node)
    }
    fn escapes_scope(&self, node: Node<'_>) -> bool {
        typescript::is_hoisted_name(node)
    }
    fn binds_past(&self, node: Node<'_>, scope_kind: &str) -> bool {
        typescript::binds_past(node, scope_kind)
    }
    fn unifies_declarations(&self, node: Node<'_>) -> bool {
        typescript::is_var_binding(node)
    }
}

struct VimscriptRules;
impl LanguageBindingRule for VimscriptRules {
    fn scope_kinds(&self) -> &'static [&'static str] {
        &[
            "function_definition",
            "lambda_expression",
            "if_statement",
            "while_loop",
            "for_loop",
            "try_statement",
        ]
    }
    fn identifier_kinds(&self) -> &'static [&'static str] {
        &["identifier", "name"]
    }
    fn declared_idents<'a>(&self, node: Node<'a>, src: &[u8]) -> Vec<Node<'a>> {
        vimscript::declared_idents(node, src)
    }
    fn is_reference(&self, node: Node<'_>) -> bool {
        vimscript::is_reference(node)
    }
    fn escapes_scope(&self, node: Node<'_>) -> bool {
        vimscript::escapes_scope(node)
    }
}

struct GoRules;
impl LanguageBindingRule for GoRules {
    fn scope_kinds(&self) -> &'static [&'static str] {
        &[
            "block",
            "function_declaration",
            "method_declaration",
            "func_literal",
            "for_statement",
            "if_statement",
            "expression_switch_statement",
            "type_switch_statement",
            "select_statement",
            "expression_case",
            "default_case",
            "type_case",
            "communication_case",
        ]
    }
    fn identifier_kinds(&self) -> &'static [&'static str] {
        &["identifier"]
    }
    fn declared_idents<'a>(&self, node: Node<'a>, src: &[u8]) -> Vec<Node<'a>> {
        go::declared_idents(node, src)
    }
    fn is_reference(&self, node: Node<'_>) -> bool {
        go::is_reference(node)
    }
}

struct JavaRules;
impl LanguageBindingRule for JavaRules {
    fn scope_kinds(&self) -> &'static [&'static str] {
        &[
            "block",
            "method_declaration",
            "constructor_declaration",
            "lambda_expression",
            "for_statement",
            "enhanced_for_statement",
            "catch_clause",
        ]
    }
    fn identifier_kinds(&self) -> &'static [&'static str] {
        &["identifier"]
    }
    fn declared_idents<'a>(&self, node: Node<'a>, src: &[u8]) -> Vec<Node<'a>> {
        java::declared_idents(node, src)
    }
    fn is_reference(&self, node: Node<'_>) -> bool {
        java::is_reference(node)
    }
}

struct SwiftRules;
impl LanguageBindingRule for SwiftRules {
    fn scope_kinds(&self) -> &'static [&'static str] {
        &[
            "function_declaration",
            "init_declaration",
            "function_body",
            "lambda_literal",
            "for_statement",
            "statements",
        ]
    }
    fn identifier_kinds(&self) -> &'static [&'static str] {
        &["simple_identifier"]
    }
    fn declared_idents<'a>(&self, node: Node<'a>, src: &[u8]) -> Vec<Node<'a>> {
        swift::declared_idents(node, src)
    }
    fn is_reference(&self, node: Node<'_>) -> bool {
        swift::is_reference(node)
    }
    fn escapes_scope(&self, node: Node<'_>) -> bool {
        swift::escapes_scope(node)
    }
}

struct CSharpRules;
impl LanguageBindingRule for CSharpRules {
    fn scope_kinds(&self) -> &'static [&'static str] {
        &[
            "block",
            "method_declaration",
            "constructor_declaration",
            "if_statement",
            "switch_statement",
            "while_statement",
            "for_statement",
            "foreach_statement",
            "catch_clause",
            "lambda_expression",
            "using_statement",
            "lock_statement",
            "checked_statement",
            "unsafe_statement",
            "fixed_statement",
        ]
    }
    fn class_scope_kinds(&self) -> &'static [&'static str] {
        &["class_declaration"]
    }
    fn identifier_kinds(&self) -> &'static [&'static str] {
        &["identifier"]
    }
    fn declared_idents<'a>(&self, node: Node<'a>, src: &[u8]) -> Vec<Node<'a>> {
        csharp::declared_idents(node, src)
    }
    fn is_reference(&self, node: Node<'_>) -> bool {
        csharp::is_reference(node)
    }
}

static RUST_RULES: RustRules = RustRules;
static CPP_RULES: CppRules = CppRules;
static PYTHON_RULES: PythonRules = PythonRules;
static GO_RULES: GoRules = GoRules;
static JAVA_RULES: JavaRules = JavaRules;
static TYPESCRIPT_RULES: TypeScriptRules = TypeScriptRules;
static CSHARP_RULES: CSharpRules = CSharpRules;
static SWIFT_RULES: SwiftRules = SwiftRules;
static VIMSCRIPT_RULES: VimscriptRules = VimscriptRules;
