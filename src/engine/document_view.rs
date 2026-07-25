// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

//! Structural selection and compact projections of indexed documents.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use anyhow::{Result, bail};

use crate::engine::document::{Document, Node, NodeKind};

#[derive(Clone, Copy)]
pub(crate) struct Selection<'a> {
    pub(crate) node: Option<&'a str>,
    pub(crate) page: Option<usize>,
    pub(crate) kind: Option<NodeKind>,
    pub(crate) depth: Option<usize>,
    pub(crate) outline: bool,
    pub(crate) overview: bool,
}

pub(crate) fn select(document: &Document, selection: Selection<'_>) -> Result<Document> {
    let index = DocumentTreeIndex::new(&document.nodes);
    if let Some(root) = selection.node {
        let Some(root_node) = index.get(root) else {
            bail!("node {root} not found");
        };
        if selection.outline && root_node.kind != NodeKind::Section {
            bail!("node {root} is not an outline node");
        }
    }
    let overview_kind = selection.overview.then(|| {
        if document
            .nodes
            .iter()
            .any(|node| node.kind == NodeKind::Section)
        {
            NodeKind::Section
        } else if document
            .nodes
            .iter()
            .any(|node| node.kind == NodeKind::StructuralSection)
        {
            NodeKind::StructuralSection
        } else {
            NodeKind::Page
        }
    });

    let mut nodes: Vec<Node> = document
        .nodes
        .iter()
        .filter(|node| {
            if let Some(root) = selection.node
                && node.id != root
                && !index.is_descendant(node, root)
            {
                return false;
            }
            if let Some(page) = selection.page {
                let node_page = node.source_anchor.as_ref().and_then(|anchor| anchor.page);
                if node_page != Some(page) {
                    return false;
                }
                if !selection.outline && node.kind == NodeKind::Section {
                    return false;
                }
            }
            if selection.outline && node.kind != NodeKind::Section {
                return false;
            }
            let kind = selection.kind.or(overview_kind);
            if kind.is_some_and(|kind| node.kind != kind) {
                return false;
            }
            true
        })
        .cloned()
        .collect();
    detach_missing_parents(&mut nodes);
    if let Some(max_depth) = selection.depth {
        let filtered_index = DocumentTreeIndex::new(&nodes);
        let depths = filtered_index.node_depths(&nodes);
        nodes.retain(|node| {
            depths
                .get(&node.id)
                .is_some_and(|depth| *depth <= max_depth)
        });
        detach_missing_parents(&mut nodes);
    }
    nodes = preorder_nodes(nodes);
    let assets = document
        .assets
        .iter()
        .filter(|asset| {
            !selection.outline
                && selection.page.is_none_or(|page| {
                    asset.source_anchor.as_ref().and_then(|anchor| anchor.page) == Some(page)
                })
        })
        .cloned()
        .collect();

    Ok(Document {
        id: document.id.clone(),
        format: document.format,
        source: document.source.clone(),
        title: document.title.clone(),
        pages: document.pages,
        nodes,
        assets,
    })
}

fn detach_missing_parents(nodes: &mut [Node]) {
    let selected_ids: HashSet<&str> = nodes.iter().map(|node| node.id.as_str()).collect();
    let detached: HashSet<String> = nodes
        .iter()
        .filter_map(|node| {
            node.parent_id
                .as_deref()
                .filter(|parent| !selected_ids.contains(parent))
                .map(|_| node.id.clone())
        })
        .collect();
    for node in nodes {
        if detached.contains(&node.id) {
            node.parent_id = None;
        }
    }
}

fn preorder_nodes(nodes: Vec<Node>) -> Vec<Node> {
    let order = {
        let mut children: HashMap<Option<&str>, Vec<usize>> = HashMap::new();
        for (index, node) in nodes.iter().enumerate() {
            children
                .entry(node.parent_id.as_deref())
                .or_default()
                .push(index);
        }

        let mut order = Vec::with_capacity(nodes.len());
        let mut visited = vec![false; nodes.len()];
        for index in children.get(&None).into_iter().flatten().copied() {
            append_preorder(index, &nodes, &children, &mut visited, &mut order);
        }
        for index in 0..nodes.len() {
            append_preorder(index, &nodes, &children, &mut visited, &mut order);
        }
        order
    };

    let mut nodes: Vec<Option<Node>> = nodes.into_iter().map(Some).collect();
    order
        .into_iter()
        .filter_map(|index| nodes[index].take())
        .collect()
}

fn append_preorder<'a>(
    index: usize,
    nodes: &'a [Node],
    children: &HashMap<Option<&'a str>, Vec<usize>>,
    visited: &mut [bool],
    order: &mut Vec<usize>,
) {
    if visited[index] {
        return;
    }
    visited[index] = true;
    order.push(index);
    if let Some(child_indices) = children.get(&Some(nodes[index].id.as_str())) {
        for child_index in child_indices {
            append_preorder(*child_index, nodes, children, visited, order);
        }
    }
}

pub(crate) fn render(document: &Document) -> String {
    let page_label = if document.pages == 1 { "page" } else { "pages" };
    let mut output = format!(
        "{} ({}, {} {page_label})\n",
        document.title,
        document.format.as_ref().to_uppercase(),
        document.pages
    );
    if document.nodes.is_empty() {
        output.push_str("(no matching nodes)");
        return output;
    }

    let index = DocumentTreeIndex::new(&document.nodes);
    let depths = index.node_depths(&document.nodes);
    let minimum_depth = depths.values().copied().min().unwrap_or_default();
    for node in &document.nodes {
        let depth = depths
            .get(&node.id)
            .copied()
            .unwrap_or_default()
            .saturating_sub(minimum_depth);
        let content = node
            .title
            .as_deref()
            .or(node.text.as_deref())
            .map(compact_text)
            .unwrap_or_default();
        let level = node
            .level
            .map(|level| format!(" {level}"))
            .unwrap_or_default();
        let page = node
            .source_anchor
            .as_ref()
            .and_then(|anchor| anchor.page)
            .map(|page| format!(" [page {page}]"))
            .unwrap_or_default();
        let separator = if content.is_empty() { "" } else { ": " };
        writeln!(
            output,
            "{}[{}] {}{level}{separator}{content}{page}",
            "  ".repeat(depth),
            node.id,
            node.kind.as_ref()
        )
        .unwrap();
    }
    output.pop();
    output
}

fn compact_text(text: &str) -> String {
    const LIMIT: usize = 500;

    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= LIMIT {
        return normalized;
    }
    let mut compact: String = normalized.chars().take(LIMIT).collect();
    compact.push_str("...");
    compact
}

pub(crate) struct DocumentTreeIndex<'a> {
    by_id: HashMap<&'a str, &'a Node>,
}

impl<'a> DocumentTreeIndex<'a> {
    pub(crate) fn new(nodes: &'a [Node]) -> Self {
        let by_id = nodes.iter().map(|node| (node.id.as_str(), node)).collect();
        Self { by_id }
    }

    pub(crate) fn get(&self, id: &str) -> Option<&'a Node> {
        self.by_id.get(id).copied()
    }

    pub(crate) fn is_descendant(&self, node: &Node, root: &str) -> bool {
        let mut visited = HashSet::new();
        let mut parent = node.parent_id.as_deref();
        while let Some(parent_id) = parent {
            if parent_id == root {
                return true;
            }
            if !visited.insert(parent_id) {
                break;
            }
            parent = self.get(parent_id).and_then(|n| n.parent_id.as_deref());
        }
        false
    }

    pub(crate) fn depth(&self, id: &str) -> usize {
        let mut depth = 0;
        let mut visited = HashSet::new();
        let mut parent = self.get(id).and_then(|n| n.parent_id.as_deref());
        while let Some(parent_id) = parent {
            if !visited.insert(parent_id) {
                break;
            }
            depth += 1;
            parent = self.get(parent_id).and_then(|n| n.parent_id.as_deref());
        }
        depth
    }

    pub(crate) fn node_depths(&self, nodes: &[Node]) -> HashMap<String, usize> {
        nodes
            .iter()
            .map(|node| (node.id.clone(), self.depth(&node.id)))
            .collect()
    }
}
