// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

//! Persistent content-addressed document indexes.

use std::fs;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};

use crate::engine::document::{
    Asset, BoundingBox, Document, DocumentFormat, Node, NodeKind, SourceAnchor,
};

const SCHEMA_VERSION: i64 = 3;

pub(crate) fn pdf_hash(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    let is_pdf = bytes.starts_with(b"%PDF-");
    is_pdf.then(|| crate::engine::hash::hash_bytes(&bytes))
}

pub(crate) fn remove_stale(
    readseek_dir: &Path,
    active_hashes: &std::collections::HashSet<String>,
) -> Result<usize> {
    let root = readseek_dir.join("documents");
    if !root.is_dir() {
        return Ok(0);
    }

    let mut removed = 0;
    for entry in fs::read_dir(&root).with_context(|| format!("read {}", root.display()))? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let Some(id) = name.to_str() else {
            continue;
        };
        if crate::engine::hash::is_hex(id, Some(64)) && !active_hashes.contains(id) {
            fs::remove_dir_all(entry.path())?;
            removed += 1;
        }
    }
    Ok(removed)
}

pub(crate) fn load(readseek_dir: &Path, id: &str) -> Result<Option<Document>> {
    let path = index_path(readseek_dir, id);
    if !path.is_file() {
        return Ok(None);
    }

    let connection = Connection::open(&path)
        .with_context(|| format!("open document index {}", path.display()))?;
    let version = schema_version(&connection)?;
    if version != SCHEMA_VERSION {
        if matches!(version, 0..=2) {
            return Ok(None);
        }
        require_schema(&connection, &path)?;
    }

    let document = connection
        .query_row(
            "SELECT id, format, pages FROM document WHERE id = ?1",
            [id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .optional()
        .with_context(|| format!("read document index {}", path.display()))?;
    let Some((id, format, pages)) = document else {
        return Ok(None);
    };
    let pages = usize::try_from(pages).context("invalid page count in document index")?;

    let nodes = load_nodes(&connection)?;
    let assets = load_assets(
        &connection,
        path.parent().context("document index path has no parent")?,
    )?;

    Ok(Some(Document {
        id,
        format: DocumentFormat::parse(&format)?,
        source: PathBuf::new(),
        title: String::new(),
        pages,
        nodes,
        assets,
    }))
}

fn load_nodes(connection: &Connection) -> Result<Vec<Node>> {
    let mut statement = connection.prepare(
        "SELECT id, parent_id, kind, title, text, page, destination,
                level, column_index, x, y, width, height
         FROM nodes ORDER BY position",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, Option<String>>(3)?,
            row.get::<_, Option<String>>(4)?,
            row.get::<_, Option<i64>>(5)?,
            row.get::<_, Option<String>>(6)?,
            row.get::<_, Option<i64>>(7)?,
            row.get::<_, Option<i64>>(8)?,
            row.get::<_, Option<f32>>(9)?,
            row.get::<_, Option<f32>>(10)?,
            row.get::<_, Option<f32>>(11)?,
            row.get::<_, Option<f32>>(12)?,
        ))
    })?;
    let mut nodes = Vec::new();
    for row in rows {
        let (
            id,
            parent_id,
            kind,
            title,
            text,
            page,
            destination,
            level,
            column,
            x,
            y,
            width,
            height,
        ) = row?;
        let page = page
            .map(usize::try_from)
            .transpose()
            .context("invalid page number in document index")?;
        let level = level
            .map(u8::try_from)
            .transpose()
            .context("invalid heading level in document index")?;
        let column = column
            .map(usize::try_from)
            .transpose()
            .context("invalid column number in document index")?;
        let bbox = load_bbox(x, y, width, height)?;
        let source_anchor =
            (page.is_some() || destination.is_some() || bbox.is_some()).then_some(SourceAnchor {
                page,
                destination,
                bbox,
            });
        nodes.push(Node {
            id,
            parent_id,
            kind: NodeKind::parse(&kind)?,
            title,
            text,
            level,
            column,
            source_anchor,
        });
    }
    Ok(nodes)
}

fn load_bbox(
    x: Option<f32>,
    y: Option<f32>,
    width: Option<f32>,
    height: Option<f32>,
) -> Result<Option<BoundingBox>> {
    match (x, y, width, height) {
        (Some(x), Some(y), Some(width), Some(height)) => Ok(Some(BoundingBox {
            x,
            y,
            width,
            height,
        })),
        (None, None, None, None) => Ok(None),
        _ => anyhow::bail!("incomplete bounding box in document index"),
    }
}

fn bbox_values(bbox: Option<BoundingBox>) -> (Option<f32>, Option<f32>, Option<f32>, Option<f32>) {
    bbox.map_or((None, None, None, None), |bbox| {
        (
            Some(bbox.x),
            Some(bbox.y),
            Some(bbox.width),
            Some(bbox.height),
        )
    })
}

fn load_assets(connection: &Connection, document_dir: &Path) -> Result<Vec<Asset>> {
    let mut statement = connection.prepare(
        "SELECT id, mime, path, page, destination, x, y, width, height
         FROM assets ORDER BY position",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, Option<i64>>(3)?,
            row.get::<_, Option<String>>(4)?,
            row.get::<_, Option<f32>>(5)?,
            row.get::<_, Option<f32>>(6)?,
            row.get::<_, Option<f32>>(7)?,
            row.get::<_, Option<f32>>(8)?,
        ))
    })?;
    let mut assets = Vec::new();
    for row in rows {
        let (id, mime, path, page, destination, x, y, width, height) = row?;
        let page = page
            .map(usize::try_from)
            .transpose()
            .context("invalid asset page number in document index")?;
        let bbox = load_bbox(x, y, width, height)?;
        let source_anchor =
            (page.is_some() || destination.is_some() || bbox.is_some()).then_some(SourceAnchor {
                page,
                destination,
                bbox,
            });
        let path = PathBuf::from(path);
        if path.is_absolute()
            || path
                .components()
                .any(|component| matches!(component, Component::ParentDir))
        {
            anyhow::bail!("invalid asset path in document index");
        }
        assets.push(Asset {
            id,
            mime,
            path: document_dir.join(path),
            source_anchor,
        });
    }
    Ok(assets)
}

pub(crate) fn store(readseek_dir: &Path, document: &Document) -> Result<()> {
    let path = index_path(readseek_dir, &document.id);
    let parent = path.parent().context("document index path has no parent")?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;

    let mut connection = Connection::open(&path)
        .with_context(|| format!("open document index {}", path.display()))?;
    initialize_schema(&connection, &path)?;
    let transaction = connection.transaction()?;
    transaction.execute("DELETE FROM assets", [])?;
    transaction.execute("DELETE FROM nodes", [])?;
    transaction.execute("DELETE FROM document", [])?;
    let pages = i64::try_from(document.pages).context("document page count is too large")?;
    transaction.execute(
        "INSERT INTO document (id, format, pages) VALUES (?1, ?2, ?3)",
        params![document.id, document.format.as_str(), pages],
    )?;
    for (position, node) in document.nodes.iter().enumerate() {
        let position = i64::try_from(position).context("too many document nodes")?;
        let (page, destination, bbox) = node
            .source_anchor
            .as_ref()
            .map_or((None, None, None), |anchor| {
                (anchor.page, anchor.destination.as_deref(), anchor.bbox)
            });
        let page = page
            .map(i64::try_from)
            .transpose()
            .context("document node page number is too large")?;
        let level = node.level.map(i64::from);
        let column = node
            .column
            .map(i64::try_from)
            .transpose()
            .context("document node column number is too large")?;
        let (x, y, width, height) = bbox_values(bbox);
        transaction.execute(
            "INSERT INTO nodes
             (id, parent_id, position, kind, title, text, page, destination,
              level, column_index, x, y, width, height)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            params![
                node.id,
                node.parent_id,
                position,
                node.kind.as_str(),
                node.title,
                node.text,
                page,
                destination,
                level,
                column,
                x,
                y,
                width,
                height,
            ],
        )?;
    }
    for (position, asset) in document.assets.iter().enumerate() {
        let position = i64::try_from(position).context("too many document assets")?;
        let (page, destination, bbox) = asset
            .source_anchor
            .as_ref()
            .map_or((None, None, None), |anchor| {
                (anchor.page, anchor.destination.as_deref(), anchor.bbox)
            });
        let page = page
            .map(i64::try_from)
            .transpose()
            .context("document asset page number is too large")?;
        let (x, y, width, height) = bbox_values(bbox);
        let asset_path = asset.path.strip_prefix(parent).with_context(|| {
            format!(
                "asset {} is outside document directory {}",
                asset.path.display(),
                parent.display()
            )
        })?;
        let asset_path = asset_path
            .to_str()
            .context("document asset path is not valid UTF-8")?;
        transaction.execute(
            "INSERT INTO assets
             (id, position, mime, path, page, destination, x, y, width, height)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                asset.id,
                position,
                asset.mime,
                asset_path,
                page,
                destination,
                x,
                y,
                width,
                height,
            ],
        )?;
    }
    transaction.commit()?;
    Ok(())
}

fn index_path(readseek_dir: &Path, id: &str) -> PathBuf {
    readseek_dir.join("documents").join(id).join("index.sqlite")
}

pub(crate) fn assets_dir(readseek_dir: &Path, id: &str) -> PathBuf {
    document_dir(readseek_dir, id).join("assets")
}

fn document_dir(readseek_dir: &Path, id: &str) -> PathBuf {
    readseek_dir.join("documents").join(id)
}

fn require_schema(connection: &Connection, path: &Path) -> Result<()> {
    let version = schema_version(connection)?;
    if version != SCHEMA_VERSION {
        anyhow::bail!(
            "unsupported document index schema {version} in {}",
            path.display()
        );
    }
    Ok(())
}

fn schema_version(connection: &Connection) -> Result<i64> {
    Ok(connection.pragma_query_value(None, "user_version", |row| row.get(0))?)
}

fn initialize_schema(connection: &Connection, path: &Path) -> Result<()> {
    connection.execute_batch("BEGIN IMMEDIATE;")?;
    let mut version = schema_version(connection)?;
    if matches!(version, 1 | 2) {
        connection.execute_batch(
            "DROP TABLE assets;
             DROP TABLE nodes;
             DROP TABLE document;
             PRAGMA user_version = 0;",
        )?;
        version = 0;
    }
    if version != 0 {
        require_schema(connection, path)?;
        connection.execute_batch("COMMIT;")?;
        return Ok(());
    }
    connection.execute_batch(
        "CREATE TABLE document (
             id TEXT PRIMARY KEY,
             format TEXT NOT NULL,
             pages INTEGER NOT NULL
         );
         CREATE TABLE nodes (
             id TEXT PRIMARY KEY,
             parent_id TEXT,
             position INTEGER NOT NULL,
             kind TEXT NOT NULL,
             title TEXT,
             text TEXT,
             page INTEGER,
             destination TEXT,
             level INTEGER,
             column_index INTEGER,
             x REAL,
             y REAL,
             width REAL,
             height REAL
         );
         CREATE INDEX nodes_parent_position ON nodes(parent_id, position);
         CREATE TABLE assets (
             id TEXT PRIMARY KEY,
             position INTEGER NOT NULL,
             mime TEXT NOT NULL,
             path TEXT NOT NULL,
             page INTEGER,
             destination TEXT,
             x REAL,
             y REAL,
             width REAL,
             height REAL
         );
         PRAGMA user_version = 3;
         ",
    )?;
    connection.execute_batch("COMMIT;")?;
    Ok(())
}
