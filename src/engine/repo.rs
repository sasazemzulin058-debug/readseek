// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

use crate::engine::flags::GitFlags;
use crate::engine::hash::LineHash;
use crate::engine::lang::{AnalysisEngine, Language};
use crate::engine::source::{SourceFile, SourceMap, Symbol};
use crate::engine::symbols;
use anyhow::{Context, Result, bail};
use rayon::prelude::*;

fn crc32_checksum(data: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(data);
    hasher.finalize()
}
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::mem::offset_of;
use std::path::{Path, PathBuf};
use std::sync::{
    OnceLock,
    atomic::{AtomicU64, Ordering},
};
use zerocopy::byteorder::{LittleEndian, U16, U32};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

const READSEEK_DIR: &str = ".readseek";
const MAPS_DIR: &str = "maps";
const DEF_INDEX_DIR: &str = "def-index";
const SHARD_COUNT: u32 = 256;
const MAGIC: [u8; 4] = *b"RSMP";
const SCHEMA_VERSION: u32 = 6;

const HEADER_SIZE: usize = size_of::<Header>();
const SYM_ENTRY_SIZE: usize = size_of::<SymEntry>();
const BLAKE3_RAW_LEN: usize = 32;
const ENGINE_TAG_NONE: u8 = 0xff;
const CHECKSUM_OFFSET: usize = offset_of!(Header, checksum);
const INDEX_MAGIC: [u8; 4] = *b"RSIX";
const INDEX_SCHEMA_VERSION: u32 = 1;
const INDEX_HEADER_SIZE: usize = size_of::<IndexHeader>();
const INDEX_NAME_ENTRY_SIZE: usize = size_of::<IndexNameEntry>();
const INDEX_PATH_ENTRY_SIZE: usize = size_of::<IndexPathEntry>();
const INDEX_CHECKSUM_OFFSET: usize = offset_of!(IndexHeader, checksum);

const _: () = assert!(
    crate::engine::hash::HASHLINE_MODULUS <= 0x10000,
    "HASHLINE_MODULUS must fit in a u16 for binary format storage"
);

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct Header {
    magic: [u8; 4],
    version: U32<LittleEndian>,
    sym_count: U32<LittleEndian>,
    strtab_sz: U32<LittleEndian>,
    file_hash: [u8; BLAKE3_RAW_LEN],
    checksum: U32<LittleEndian>,
    lang_tag: U16<LittleEndian>,
    engine_tag: u8,
    _reserved: [u8; 9],
}

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct SymEntry {
    kind_off: U32<LittleEndian>,
    name_off: U32<LittleEndian>,
    qname_off: U32<LittleEndian>,
    start_line: U32<LittleEndian>,
    end_line: U32<LittleEndian>,
    start_byte: U32<LittleEndian>,
    end_byte: U32<LittleEndian>,
    name_byte: U32<LittleEndian>,
    kind_len: U16<LittleEndian>,
    name_len: U16<LittleEndian>,
    qname_len: U16<LittleEndian>,
    start_hash: U16<LittleEndian>,
    end_hash: U16<LittleEndian>,
}

const _: () = assert!(size_of::<Header>() == 64, "Header must be exactly 64 bytes");
const _: () = assert!(
    size_of::<SymEntry>() == 42,
    "SymEntry must be exactly 42 bytes",
);

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct IndexHeader {
    magic: [u8; 4],
    version: U32<LittleEndian>,
    name_count: U32<LittleEndian>,
    path_count: U32<LittleEndian>,
    strtab_sz: U32<LittleEndian>,
    checksum: U32<LittleEndian>,
}

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct IndexNameEntry {
    name_off: U32<LittleEndian>,
    name_len: U16<LittleEndian>,
    first_path: U32<LittleEndian>,
    path_count: U16<LittleEndian>,
}

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct IndexPathEntry {
    path_off: U32<LittleEndian>,
    path_len: U16<LittleEndian>,
}

const _: () = assert!(
    size_of::<IndexHeader>() == 24,
    "IndexHeader must be exactly 24 bytes"
);
const _: () = assert!(
    size_of::<IndexNameEntry>() == 12,
    "IndexNameEntry must be exactly 12 bytes"
);
const _: () = assert!(
    size_of::<IndexPathEntry>() == 6,
    "IndexPathEntry must be exactly 6 bytes"
);

#[derive(Debug, serde::Serialize)]
pub(crate) struct UpdateStats {
    pub(crate) created: usize,
    pub(crate) removed: usize,
    pub(crate) unchanged: usize,
}

pub(crate) struct DefIndexEntry {
    pub(crate) name: String,
    pub(crate) qualified_name: String,
    pub(crate) path: PathBuf,
}

struct UpdatePathResult {
    file_hash: String,
    created: bool,
    entries: Vec<DefIndexEntry>,
}

static DIR_OVERRIDE: OnceLock<PathBuf> = OnceLock::new();
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

/// Pin the `.readseek` directory, bypassing ancestor discovery (`--readseek-dir`).
pub(crate) fn set_dir_override(path: PathBuf) {
    DIR_OVERRIDE.set(path).ok();
}

fn dir_override() -> Option<&'static Path> {
    DIR_OVERRIDE.get().map(PathBuf::as_path)
}

pub(crate) fn find_readseek_dir(base: &Path) -> Option<PathBuf> {
    if let Some(dir) = dir_override() {
        return dir.is_dir().then(|| dir.to_path_buf());
    }
    let base = base.canonicalize().ok()?;
    base.ancestors()
        .find(|ancestor| {
            let candidate = ancestor.join(READSEEK_DIR);
            candidate.is_dir()
        })
        .map(|ancestor| ancestor.join(READSEEK_DIR))
}

fn readseek_dir_or_err(base: &Path) -> Result<PathBuf> {
    find_readseek_dir(base)
        .with_context(|| format!("no {READSEEK_DIR} directory found; run 'readseek init' first"))
}

#[derive(Clone, Debug)]
pub(crate) struct InitResult {
    pub(crate) path: PathBuf,
    pub(crate) reinitialized: bool,
}

pub(crate) fn init(dir: &Path) -> Result<InitResult> {
    let canonical = dir.canonicalize().context("resolve init path")?;
    let (readseek_dir, canonical_readseek) = dir_override().map_or_else(
        || (dir.join(READSEEK_DIR), canonical.join(READSEEK_DIR)),
        |dir| (dir.to_path_buf(), dir.to_path_buf()),
    );
    let reinitialized = canonical_readseek.exists();
    let maps_dir = canonical_readseek.join(MAPS_DIR);
    fs::create_dir_all(&maps_dir).with_context(|| format!("create {}", maps_dir.display()))?;
    let def_index_dir = canonical_readseek.join(DEF_INDEX_DIR);
    fs::create_dir_all(&def_index_dir)
        .with_context(|| format!("create {}", def_index_dir.display()))?;

    Ok(InitResult {
        path: readseek_dir,
        reinitialized,
    })
}

fn hex_hash_to_raw(hex_str: &str) -> Result<[u8; BLAKE3_RAW_LEN]> {
    crate::engine::hash::decode_hex(hex_str).with_context(|| format!("invalid hex hash: {hex_str}"))
}

fn map_path(readseek_dir: &Path, hash_hex: &str) -> PathBuf {
    readseek_dir
        .join(MAPS_DIR)
        .join(&hash_hex[..2])
        .join(&hash_hex[2..])
}

fn shard_bucket(name: &str) -> u32 {
    xxhash_rust::xxh32::xxh32(name.as_bytes(), 0) % SHARD_COUNT
}

fn def_index_shard_path(readseek_dir: &Path, name: &str) -> PathBuf {
    readseek_dir
        .join(DEF_INDEX_DIR)
        .join(format!("{}.bin", shard_bucket(name)))
}

pub(crate) fn load_map(
    readseek_dir: &Path,
    file_hash: &str,
) -> Result<Option<(SourceMap, Language, Option<AnalysisEngine>)>> {
    let path = map_path(readseek_dir, file_hash);
    if !path.exists() {
        return Ok(None);
    }

    let data = fs::read(&path).with_context(|| format!("read {}", path.display()))?;

    if data.len() < HEADER_SIZE {
        tracing::debug!(target: "tracing", "truncated map file: {}", path.display());
        return Ok(None);
    }

    let header = Header::ref_from_bytes(&data[..HEADER_SIZE])
        .map_err(|e| anyhow::anyhow!("parse header of {}: {e}", path.display()))?;

    if header.magic != MAGIC {
        tracing::debug!(target: "tracing", "invalid magic in {}", path.display());
        return Ok(None);
    }
    if header.version.get() != SCHEMA_VERSION {
        tracing::debug!(
            target: "tracing",
            "unsupported schema version {} in {}",
            header.version.get(),
            path.display()
        );
        return Ok(None);
    }

    let expected_hash = hex_hash_to_raw(file_hash)?;
    if header.file_hash != expected_hash {
        tracing::debug!(target: "tracing", "hash mismatch in {}", path.display());
        return Ok(None);
    }

    let computed = crc32_checksum(&data[HEADER_SIZE..]);
    if header.checksum.get() != computed {
        tracing::debug!(target: "tracing", "checksum mismatch in {}", path.display());
        return Ok(None);
    }

    let language = Language::from_repr(header.lang_tag.get())
        .with_context(|| format!("unknown language tag {}", header.lang_tag.get()))?;
    let engine = if header.engine_tag == ENGINE_TAG_NONE {
        None
    } else {
        Some(
            AnalysisEngine::from_repr(header.engine_tag)
                .with_context(|| format!("unknown analysis engine tag {}", header.engine_tag))?,
        )
    };

    let sym_count = header.sym_count.get() as usize;
    if sym_count == 0 {
        return Ok(Some((
            SourceMap {
                symbols: Vec::new(),
            },
            language,
            engine,
        )));
    }

    let strtab_sz = header.strtab_sz.get() as usize;

    let sym_total = sym_count
        .checked_mul(SYM_ENTRY_SIZE)
        .context("sym_count overflow")?;
    let expected = HEADER_SIZE
        .checked_add(sym_total)
        .and_then(|v| v.checked_add(strtab_sz))
        .context("map size overflow")?;
    if data.len() != expected {
        bail!(
            "invalid map: buffer is {} bytes, header claims {}",
            data.len(),
            expected
        );
    }

    let strtab_start = HEADER_SIZE
        .checked_add(sym_total)
        .context("map symbol table size overflow")?;
    let strtab_end = strtab_start
        .checked_add(strtab_sz)
        .context("map string table size overflow")?;

    if data.len() < strtab_end {
        tracing::debug!(target: "tracing", "truncated data in {}", path.display());
        return Ok(None);
    }

    let sym_bytes = &data[HEADER_SIZE..strtab_start];
    let strtab = &data[strtab_start..strtab_end];

    let symbols = (0..sym_count)
        .map(|i| parse_sym_entry(sym_bytes, strtab, i, &path))
        .collect::<Result<Vec<_>>>()?;

    Ok(Some((SourceMap { symbols }, language, engine)))
}

fn parse_sym_entry(sym_bytes: &[u8], strtab: &[u8], i: usize, path: &Path) -> Result<Symbol> {
    let start = i
        .checked_mul(SYM_ENTRY_SIZE)
        .context("symbol index overflow")?;
    let end = start
        .checked_add(SYM_ENTRY_SIZE)
        .context("symbol entry range overflow")?;
    let entry_bytes = sym_bytes
        .get(start..end)
        .with_context(|| format!("symbol entry {i} out of bounds in {}", path.display()))?;
    let entry = SymEntry::ref_from_bytes(entry_bytes)
        .map_err(|e| anyhow::anyhow!("parse sym entry {i} of {}: {e}", path.display()))?;
    let kind = read_str(strtab, entry.kind_off.get(), entry.kind_len.get())?;
    let name = read_str(strtab, entry.name_off.get(), entry.name_len.get())?;
    let qualified_name = read_str(strtab, entry.qname_off.get(), entry.qname_len.get())?;
    Ok(Symbol {
        kind: kind.to_owned(),
        name: name.to_owned(),
        qualified_name: qualified_name.to_owned(),
        start_line: entry.start_line.get() as usize,
        end_line: entry.end_line.get() as usize,
        start_hash: LineHash::try_from(entry.start_hash.get())?,
        end_hash: LineHash::try_from(entry.end_hash.get())?,
        start_byte: entry.start_byte.get() as usize,
        end_byte: entry.end_byte.get() as usize,
        name_byte: entry.name_byte.get() as usize,
    })
}

fn read_str(strtab: &[u8], offset: u32, len: u16) -> Result<&str> {
    let start = usize::try_from(offset).context("string table offset overflow")?;
    let len = usize::from(len);
    let end = start
        .checked_add(len)
        .context("string table range overflow")?;
    let bytes = strtab.get(start..end).with_context(|| {
        format!(
            "string table out of bounds: offset={offset} len={len} strtab_len={}",
            strtab.len()
        )
    })?;
    std::str::from_utf8(bytes)
        .with_context(|| format!("invalid UTF-8 in string table at offset {offset}"))
}

pub(crate) fn store_map(
    readseek_dir: &Path,
    file_hash: &str,
    source: &SourceFile,
    source_map: &SourceMap,
) -> Result<()> {
    let language = source.detection.language;
    let engine_tag = source.detection.engine.map_or(ENGINE_TAG_NONE, u8::from);

    let raw_hash = hex_hash_to_raw(file_hash)?;
    let sym_count = u32::try_from(source_map.symbols.len())
        .with_context(|| format!("too many symbols: {}", source_map.symbols.len()))?;

    let string_bytes = source_map.symbols.iter().fold(0usize, |total, symbol| {
        total
            .saturating_add(symbol.kind.len())
            .saturating_add(symbol.name.len())
            .saturating_add(symbol.qualified_name.len())
    });
    let mut strtab = Vec::with_capacity(string_bytes);
    let mut entries = Vec::with_capacity(source_map.symbols.len());

    for symbol in &source_map.symbols {
        let kind_off = u32::try_from(strtab.len())?;
        let kind_len = u16::try_from(symbol.kind.len())
            .with_context(|| format!("kind name too long: {}", symbol.kind.len()))?;
        strtab.extend_from_slice(symbol.kind.as_bytes());

        let name_off = u32::try_from(strtab.len())?;
        let name_len = u16::try_from(symbol.name.len())
            .with_context(|| format!("name too long: {}", symbol.name.len()))?;
        strtab.extend_from_slice(symbol.name.as_bytes());

        let qname_off = u32::try_from(strtab.len())?;
        let qname_len = u16::try_from(symbol.qualified_name.len())
            .with_context(|| format!("qualified name too long: {}", symbol.qualified_name.len()))?;
        strtab.extend_from_slice(symbol.qualified_name.as_bytes());

        entries.push(SymEntry {
            kind_off: U32::new(kind_off),
            name_off: U32::new(name_off),
            qname_off: U32::new(qname_off),
            start_line: U32::new(u32::try_from(symbol.start_line)?),
            end_line: U32::new(u32::try_from(symbol.end_line)?),
            start_byte: U32::new(u32::try_from(symbol.start_byte)?),
            end_byte: U32::new(u32::try_from(symbol.end_byte)?),
            name_byte: U32::new(u32::try_from(symbol.name_byte)?),
            kind_len: U16::new(kind_len),
            name_len: U16::new(name_len),
            qname_len: U16::new(qname_len),
            start_hash: U16::new(u16::from(symbol.start_hash)),
            end_hash: U16::new(u16::from(symbol.end_hash)),
        });
    }

    let strtab_sz = u32::try_from(strtab.len())?;

    let header = Header {
        magic: MAGIC,
        version: U32::new(SCHEMA_VERSION),
        lang_tag: U16::new(u16::from(language)),
        engine_tag,
        _reserved: [0u8; 9],
        sym_count: U32::new(sym_count),
        strtab_sz: U32::new(strtab_sz),
        file_hash: raw_hash,
        checksum: U32::new(0),
    };

    let sym_total = entries
        .len()
        .checked_mul(SYM_ENTRY_SIZE)
        .context("symbol table size overflow")?;
    let total_size = HEADER_SIZE
        .checked_add(sym_total)
        .and_then(|size| size.checked_add(strtab.len()))
        .context("map size overflow")?;
    let mut buf = Vec::with_capacity(total_size);
    buf.extend_from_slice(header.as_bytes());
    for entry in &entries {
        buf.extend_from_slice(entry.as_bytes());
    }
    buf.extend_from_slice(&strtab);
    let checksum = crc32_checksum(&buf[HEADER_SIZE..]);
    buf[CHECKSUM_OFFSET..CHECKSUM_OFFSET + size_of::<U32<LittleEndian>>()]
        .copy_from_slice(&checksum.to_le_bytes());

    let path = map_path(readseek_dir, file_hash);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }

    write_atomic(&path, &buf).with_context(|| format!("write {}", path.display()))?;

    Ok(())
}

pub(crate) fn load_index(readseek_dir: &Path, name: &str) -> Result<Option<Vec<PathBuf>>> {
    let path = def_index_shard_path(readseek_dir, name);
    if !path.exists() {
        return Ok(None);
    }
    let data = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    if data.len() < INDEX_HEADER_SIZE {
        tracing::debug!(target: "tracing", "truncated def-index file: {}", path.display());
        return Ok(None);
    }
    let header = IndexHeader::ref_from_bytes(&data[..INDEX_HEADER_SIZE])
        .map_err(|e| anyhow::anyhow!("parse def-index header of {}: {e}", path.display()))?;
    if header.magic != INDEX_MAGIC {
        tracing::debug!(target: "tracing", "invalid magic in {}", path.display());
        return Ok(None);
    }
    if header.version.get() != INDEX_SCHEMA_VERSION {
        tracing::debug!(
            target: "tracing",
            "unsupported def-index schema version {} in {}",
            header.version.get(),
            path.display()
        );
        return Ok(None);
    }
    if header.checksum.get() != crc32_checksum(&data[INDEX_HEADER_SIZE..]) {
        tracing::debug!(target: "tracing", "checksum mismatch in {}", path.display());
        return Ok(None);
    }
    let name_count = header.name_count.get() as usize;
    let path_count = header.path_count.get() as usize;
    let strtab_sz = header.strtab_sz.get() as usize;
    let name_table_size = name_count
        .checked_mul(INDEX_NAME_ENTRY_SIZE)
        .context("def-index name table overflow")?;
    let path_table_size = path_count
        .checked_mul(INDEX_PATH_ENTRY_SIZE)
        .context("def-index path table overflow")?;
    let name_start = INDEX_HEADER_SIZE;
    let name_end = name_start
        .checked_add(name_table_size)
        .context("name end overflow")?;
    let path_end = name_end
        .checked_add(path_table_size)
        .context("path end overflow")?;
    let strtab_end = path_end
        .checked_add(strtab_sz)
        .context("strtab end overflow")?;
    if data.len() < strtab_end {
        tracing::debug!(target: "tracing", "truncated def-index data in {}", path.display());
        return Ok(None);
    }
    let name_bytes = &data[name_start..name_end];
    let path_bytes = &data[name_end..path_end];
    let strtab = &data[path_end..strtab_end];
    let needle = name.as_bytes();
    let mut lo = 0usize;
    let mut hi = name_count;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let entry = IndexNameEntry::ref_from_bytes(
            &name_bytes
                [mid * INDEX_NAME_ENTRY_SIZE..mid * INDEX_NAME_ENTRY_SIZE + INDEX_NAME_ENTRY_SIZE],
        )
        .map_err(|e| anyhow::anyhow!("parse def-index name entry {mid}: {e}"))?;
        let entry_name = read_str(strtab, entry.name_off.get(), entry.name_len.get())?;
        match needle.cmp(entry_name.as_bytes()) {
            std::cmp::Ordering::Less => hi = mid,
            std::cmp::Ordering::Greater => lo = mid + 1,
            std::cmp::Ordering::Equal => {
                let first = entry.first_path.get() as usize;
                let count = entry.path_count.get() as usize;
                let Some(end) = first.checked_add(count) else {
                    tracing::debug!(target: "tracing", "invalid def-index path range in {}", path.display());
                    return Ok(None);
                };
                if end > path_count {
                    tracing::debug!(target: "tracing", "invalid def-index path range in {}", path.display());
                    return Ok(None);
                }
                let mut paths = Vec::with_capacity(count);
                for i in 0..count {
                    let idx = first + i;
                    let pe = IndexPathEntry::ref_from_bytes(
                        &path_bytes[idx * INDEX_PATH_ENTRY_SIZE
                            ..idx * INDEX_PATH_ENTRY_SIZE + INDEX_PATH_ENTRY_SIZE],
                    )
                    .map_err(|e| anyhow::anyhow!("parse def-index path entry {idx}: {e}"))?;
                    let path_str = read_str(strtab, pe.path_off.get(), pe.path_len.get())?;
                    paths.push(PathBuf::from(path_str));
                }
                return Ok(Some(paths));
            }
        }
    }
    Ok(Some(Vec::new()))
}

pub(crate) fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let dir = path.parent().context("map path has no parent")?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pid = std::process::id();

    for _ in 0..64 {
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let tmp = dir.join(format!(".tmp-{pid}-{timestamp:x}-{sequence:x}"));
        let file = match OpenOptions::new().write(true).create_new(true).open(&tmp) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error).with_context(|| format!("create {}", tmp.display())),
        };
        let write_result = {
            let mut file = file;
            file.write_all(data)
        };
        if let Err(error) = write_result {
            let _ = fs::remove_file(&tmp);
            return Err(error).with_context(|| format!("write {}", tmp.display()));
        }
        return fs::rename(&tmp, path)
            .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()));
    }

    bail!(
        "could not create a unique temporary file in {}",
        dir.display()
    )
}

pub(crate) fn update(dir: &Path, flags: GitFlags) -> Result<UpdateStats> {
    let readseek_dir = readseek_dir_or_err(dir)?;
    // With an override the readseek dir is decoupled from the work tree, so the
    // project root is the caller-supplied work tree rather than the dir's parent.
    let project_root = match dir_override() {
        Some(_) => dir
            .canonicalize()
            .with_context(|| format!("resolve {}", dir.display()))?,
        None => readseek_dir
            .parent()
            .context(".readseek has no parent")?
            .to_path_buf(),
    };
    let paths = crate::engine::paths::command_paths(&project_root, flags)?;

    let results = paths
        .par_iter()
        .map(|path| process_update_path(&readseek_dir, path))
        .collect::<Result<Vec<_>>>()?;

    let mut active_hashes = HashSet::with_capacity(results.len());
    let mut index_entries = Vec::with_capacity(results.len());
    let mut stats = UpdateStats {
        created: 0,
        removed: 0,
        unchanged: 0,
    };

    for result in results.into_iter().flatten() {
        active_hashes.insert(result.file_hash);
        index_entries.extend(result.entries);
        if result.created {
            stats.created += 1;
        } else {
            stats.unchanged += 1;
        }
    }

    let shards = build_index_shards(index_entries);
    let written_prefixes = write_index_shards(&readseek_dir, &shards)?;
    remove_stale_index_shards(&readseek_dir, &written_prefixes)?;
    stats.removed += remove_stale_maps(&readseek_dir, &active_hashes)?;
    let active_document_hashes: HashSet<String> = paths
        .par_iter()
        .filter_map(|path| crate::engine::document_store::pdf_hash(path))
        .collect();
    stats.removed +=
        crate::engine::document_store::remove_stale(&readseek_dir, &active_document_hashes)?;

    let active_image_hashes: HashSet<String> = paths
        .par_iter()
        .filter_map(|path| crate::engine::vision_cache::image_hash(path))
        .collect();
    stats.removed +=
        crate::engine::vision_cache::remove_stale(&readseek_dir, &active_image_hashes)?;

    Ok(stats)
}

fn process_update_path(readseek_dir: &Path, path: &Path) -> Result<Option<UpdatePathResult>> {
    let Some(source) = crate::engine::source::load_indexable_source(path, None)? else {
        return Ok(None);
    };
    if !source.detection.supported {
        return Ok(None);
    }

    let cached = load_map(readseek_dir, &source.file_hash)?
        .filter(|(_, language, _)| *language == source.detection.language);
    let (source_map, created) = if let Some((source_map, _, _)) = cached {
        (source_map, false)
    } else {
        let source_map = symbols::parse_source_map(&source)?;
        store_map(readseek_dir, &source.file_hash, &source, &source_map)?;
        (source_map, true)
    };

    let entries = source_map
        .symbols
        .into_iter()
        .map(|symbol| DefIndexEntry {
            name: symbol.name,
            qualified_name: symbol.qualified_name,
            path: source.path.clone(),
        })
        .collect();

    Ok(Some(UpdatePathResult {
        file_hash: source.file_hash,
        created,
        entries,
    }))
}

fn build_index_shards(
    index_entries: Vec<DefIndexEntry>,
) -> BTreeMap<u32, BTreeMap<String, Vec<PathBuf>>> {
    let mut shards: BTreeMap<u32, BTreeMap<String, Vec<PathBuf>>> = BTreeMap::new();
    for entry in index_entries {
        let bucket = shard_bucket(&entry.name);
        shards
            .entry(bucket)
            .or_default()
            .entry(entry.name.clone())
            .or_default()
            .push(entry.path.clone());
        if entry.qualified_name != entry.name {
            let bucket = shard_bucket(&entry.qualified_name);
            shards
                .entry(bucket)
                .or_default()
                .entry(entry.qualified_name.clone())
                .or_default()
                .push(entry.path);
        }
    }
    for inner in shards.values_mut() {
        for paths in inner.values_mut() {
            paths.sort();
            paths.dedup();
        }
    }
    shards
}

fn write_index_shards(
    readseek_dir: &Path,
    shards: &BTreeMap<u32, BTreeMap<String, Vec<PathBuf>>>,
) -> Result<BTreeSet<u32>> {
    let shard_dir = readseek_dir.join(DEF_INDEX_DIR);
    fs::create_dir_all(&shard_dir).with_context(|| format!("create {}", shard_dir.display()))?;
    let mut written_buckets = BTreeSet::new();
    for (bucket, shard) in shards {
        let data = serialize_shard(shard)?;
        let path = shard_dir.join(format!("{bucket}.bin"));
        write_atomic(&path, &data)?;
        written_buckets.insert(*bucket);
    }
    Ok(written_buckets)
}

fn serialize_shard(shard: &BTreeMap<String, Vec<PathBuf>>) -> Result<Vec<u8>> {
    let mut strtab = Vec::new();
    let mut name_entries = Vec::new();
    let mut path_entries = Vec::new();

    for (name, paths) in shard {
        let name_off = u32::try_from(strtab.len()).context("strtab overflow")?;
        let name_len =
            u16::try_from(name.len()).with_context(|| format!("name too long: {}", name.len()))?;
        strtab.extend_from_slice(name.as_bytes());

        let first_path = u32::try_from(path_entries.len()).context("path table overflow")?;
        let path_count = u16::try_from(paths.len())
            .with_context(|| format!("too many paths for one name: {}", paths.len()))?;
        for path in paths {
            let path_str = path.to_string_lossy();
            let path_off = u32::try_from(strtab.len()).context("strtab overflow")?;
            let path_len = u16::try_from(path_str.len())
                .with_context(|| format!("path too long: {}", path_str.len()))?;
            strtab.extend_from_slice(path_str.as_bytes());
            path_entries.push(IndexPathEntry {
                path_off: U32::new(path_off),
                path_len: U16::new(path_len),
            });
        }
        name_entries.push(IndexNameEntry {
            name_off: U32::new(name_off),
            name_len: U16::new(name_len),
            first_path: U32::new(first_path),
            path_count: U16::new(path_count),
        });
    }

    let name_count = u32::try_from(name_entries.len()).context("too many names")?;
    let path_count = u32::try_from(path_entries.len()).context("too many paths")?;
    let strtab_sz = u32::try_from(strtab.len()).context("strtab size overflow")?;

    let header = IndexHeader {
        magic: INDEX_MAGIC,
        version: U32::new(INDEX_SCHEMA_VERSION),
        name_count: U32::new(name_count),
        path_count: U32::new(path_count),
        strtab_sz: U32::new(strtab_sz),
        checksum: U32::new(0),
    };

    let name_total = name_entries
        .len()
        .checked_mul(INDEX_NAME_ENTRY_SIZE)
        .context("name table overflow")?;
    let path_total = path_entries
        .len()
        .checked_mul(INDEX_PATH_ENTRY_SIZE)
        .context("path table overflow")?;
    let total = INDEX_HEADER_SIZE
        .checked_add(name_total)
        .and_then(|v| v.checked_add(path_total))
        .and_then(|v| v.checked_add(strtab.len()))
        .context("shard size overflow")?;

    let mut buf = Vec::with_capacity(total);
    buf.extend_from_slice(header.as_bytes());
    for entry in &name_entries {
        buf.extend_from_slice(entry.as_bytes());
    }
    for entry in &path_entries {
        buf.extend_from_slice(entry.as_bytes());
    }
    buf.extend_from_slice(&strtab);

    let checksum = crc32_checksum(&buf[INDEX_HEADER_SIZE..]);
    buf[INDEX_CHECKSUM_OFFSET..INDEX_CHECKSUM_OFFSET + size_of::<U32<LittleEndian>>()]
        .copy_from_slice(&checksum.to_le_bytes());

    Ok(buf)
}

fn remove_stale_index_shards(readseek_dir: &Path, written_buckets: &BTreeSet<u32>) -> Result<()> {
    let shard_dir = readseek_dir.join(DEF_INDEX_DIR);
    for entry in
        fs::read_dir(&shard_dir).with_context(|| format!("read {}", shard_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "bin") {
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            let retained = stem
                .parse::<u32>()
                .is_ok_and(|bucket| written_buckets.contains(&bucket));
            if !retained {
                fs::remove_file(&path)?;
            }
        }
    }
    Ok(())
}

fn remove_stale_maps(readseek_dir: &Path, active_hashes: &HashSet<String>) -> Result<usize> {
    let mut removed = 0;
    let maps_root = readseek_dir.join(MAPS_DIR);
    if maps_root.is_dir() {
        for entry in
            fs::read_dir(&maps_root).with_context(|| format!("read {}", maps_root.display()))?
        {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                for file_entry in fs::read_dir(entry.path())? {
                    let file_entry = file_entry?;
                    let filename = file_entry.file_name();
                    let hash_fragment = filename.to_string_lossy();
                    let parent = file_entry
                        .path()
                        .parent()
                        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()));
                    if let Some(prefix) = parent {
                        let hash_hex = format!("{prefix}{hash_fragment}");
                        if crate::engine::hash::is_hex(&hash_hex, Some(BLAKE3_RAW_LEN * 2))
                            && !active_hashes.contains(&hash_hex)
                        {
                            fs::remove_file(file_entry.path())?;
                            removed += 1;
                        }
                    }
                }
            }
        }
    }

    Ok(removed)
}
