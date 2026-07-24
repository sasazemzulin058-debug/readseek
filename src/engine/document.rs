// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

//! Format-neutral indexed document model.

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use strum_macros::{AsRefStr, Display, EnumString};

#[derive(Clone, Copy, Debug, Deserialize, Display, EnumString, AsRefStr, Serialize)]
#[strum(serialize_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub(crate) enum DocumentFormat {
    Pdf,
}

impl DocumentFormat {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        value
            .parse()
            .map_err(|_| anyhow!("unsupported indexed document format: {value}"))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct Document {
    pub(crate) id: String,
    pub(crate) format: DocumentFormat,
    pub(crate) source: PathBuf,
    pub(crate) title: String,
    pub(crate) pages: usize,
    pub(crate) nodes: Vec<Node>,
    pub(crate) assets: Vec<Asset>,
}

impl Document {
    pub(crate) fn rebind_source(&mut self, path: &Path) {
        self.source = path.to_path_buf();
        path.file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("document")
            .clone_into(&mut self.title);
    }
}

#[derive(
    Clone, Copy, Debug, Deserialize, Display, EnumString, AsRefStr, Eq, PartialEq, Serialize,
)]
#[strum(serialize_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub(crate) enum NodeKind {
    Artifact,
    Footer,
    Header,
    Heading,
    MarginalLabel,
    Page,
    PageNumber,
    Paragraph,
    Section,
    StructuralSection,
}

impl NodeKind {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        value
            .parse()
            .map_err(|_| anyhow!("unsupported indexed node kind: {value}"))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct Node {
    pub(crate) id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) parent_id: Option<String>,
    pub(crate) kind: NodeKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) level: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) column: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) source_anchor: Option<SourceAnchor>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct SourceAnchor {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) page: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) destination: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) bbox: Option<BoundingBox>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub(crate) struct BoundingBox {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) width: f32,
    pub(crate) height: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct Asset {
    pub(crate) id: String,
    pub(crate) mime: String,
    pub(crate) path: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) source_anchor: Option<SourceAnchor>,
}
