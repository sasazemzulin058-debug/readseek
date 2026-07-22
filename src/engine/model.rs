// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

//! Qwen3-VL model cache. The quantized text model and multimodal projector are
//! downloaded and SHA-256-verified on first use, then reused by file size.

use std::fs;
use std::io::{BufReader, IsTerminal as _, Read as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, UNIX_EPOCH};

use anyhow::{Context as _, Result, anyhow};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const BASE: &str = "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/52d6c8ffea26cc873ac5ad116f8631268d7eb503";
const CACHE_SUBDIR: &str = "models";
const LOCK_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const VERIFIED_SCHEMA_VERSION: u32 = 1;

/// `(remote path, local file name, byte size, sha256)`.
const FILES: &[(&str, &str, u64, &str)] = &[
    (
        "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
        "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
        1_107_409_952,
        "089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae",
    ),
    (
        "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
        "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
        445_053_216,
        "f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82",
    ),
];

#[derive(Deserialize, Serialize)]
struct VerifiedFile {
    schema_version: u32,
    sha256: String,
    size: u64,
    modified_ns: u128,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct FileStamp {
    size: u64,
    modified_ns: u128,
}

/// Return a cached model file, downloading and verifying it when absent.
pub(crate) fn file(name: &str) -> Result<PathBuf> {
    let (remote, local, size, sha256) = FILES
        .iter()
        .find(|(_, local, _, _)| *local == name)
        .copied()
        .ok_or_else(|| anyhow!("unknown model file `{name}`"))?;
    let dir = cache_dir()?.join(CACHE_SUBDIR);
    fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    let target = dir.join(local);
    let _lock = ModelLock::acquire(&target.with_extension("file-lock"))?;
    if let Err(error) = remove_stale_parts(&target) {
        tracing::debug!(target: "tracing", "stale model download cleanup skipped: {error:#}");
    }
    if valid_cached_file(&target, size, sha256) {
        return Ok(target);
    }
    let _ = fs::remove_file(verified_path(&target));
    let _ = fs::remove_file(&target);

    download(remote, local, &target)?;
    Ok(target)
}

fn cache_dir() -> Result<PathBuf> {
    let dir = dirs::cache_dir()
        .context("no user cache directory is available on this platform")?
        .join("readseek");
    fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    Ok(dir)
}

/// Download a verified model through a unique temporary file.
fn download(remote: &str, local: &str, target: &PathBuf) -> Result<()> {
    let url = format!("{BASE}/{remote}");
    let part = unique_part(target)?;
    let agent = ureq::Agent::with_parts(
        ureq::config::Config::default(),
        ureq::unversioned::transport::DefaultConnector::default(),
        crate::engine::resolver::DohResolver,
    );
    let response = agent
        .get(&url)
        .call()
        .with_context(|| format!("download {url}"))?;
    let total = response.body().content_length();

    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&part)
        .with_context(|| format!("create {}", part.display()))?;
    let mut body = response.into_body();
    let mut reader = body.as_reader();

    if std::io::stdout().is_terminal() {
        let progress = MultiProgress::new();
        let bar = match total {
            Some(length) => ProgressBar::new(length),
            None => ProgressBar::new_spinner(),
        };
        bar.set_style(
            ProgressStyle::with_template(
                "{prefix:<36} {bar:30} {bytes}/{total_bytes} ({bytes_per_sec}, {eta})",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("=> "),
        );
        bar.set_prefix(local.to_owned());
        let bar = progress.add(bar);
        let mut wrapped = bar.wrap_read(&mut reader);
        std::io::copy(&mut wrapped, &mut file).with_context(|| format!("read {url}"))?;
        bar.finish_with_message(format!("{local} done"));
    } else {
        std::io::copy(&mut reader, &mut file).with_context(|| format!("read {url}"))?;
    }
    file.sync_all()
        .with_context(|| format!("sync {}", part.display()))?;
    drop(file);
    let (_, _, expected_size, expected_sha256) = FILES
        .iter()
        .find(|(_, name, _, _)| *name == local)
        .copied()
        .expect("download only receives known model files");
    let actual_size = fs::metadata(&part)
        .with_context(|| format!("stat {}", part.display()))?
        .len();
    let actual_sha256 = sha256_file(&part)?;
    if actual_size != expected_size || actual_sha256 != expected_sha256 {
        let _ = fs::remove_file(&part);
        return Err(anyhow!(
            "model verification failed for {local}: expected {expected_size} bytes and {expected_sha256}, got {actual_size} bytes and {actual_sha256}"
        ));
    }
    fs::rename(&part, target)
        .with_context(|| format!("rename {} -> {}", part.display(), target.display()))?;
    if let Err(error) = write_verified(target, expected_size, expected_sha256) {
        tracing::debug!(target: "tracing", "model verification marker {}: {error:#}", target.display());
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn valid_cached_file(path: &Path, expected_size: u64, expected_sha256: &str) -> bool {
    let Some(before) = file_stamp(path) else {
        return false;
    };
    if before.size != expected_size {
        return false;
    }
    let marker = verified_path(path);
    let verified = fs::read(&marker)
        .ok()
        .and_then(|data| serde_json::from_slice::<VerifiedFile>(&data).ok());
    if verified.is_some_and(|verified| {
        verified.schema_version == VERIFIED_SCHEMA_VERSION
            && verified.sha256 == expected_sha256
            && verified.size == before.size
            && verified.modified_ns == before.modified_ns
    }) {
        return true;
    }

    let Ok(actual_sha256) = sha256_file(path) else {
        return false;
    };
    let Some(after) = file_stamp(path) else {
        return false;
    };
    if before != after || actual_sha256 != expected_sha256 {
        return false;
    }
    if let Err(error) = write_verified(path, expected_size, expected_sha256) {
        tracing::debug!(target: "tracing", "model verification marker {}: {error:#}", marker.display());
    }
    true
}

fn file_stamp(path: &Path) -> Option<FileStamp> {
    let metadata = fs::metadata(path).ok()?;
    if !metadata.is_file() {
        return None;
    }
    let modified_ns = metadata
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_nanos();
    Some(FileStamp {
        size: metadata.len(),
        modified_ns,
    })
}

fn verified_path(path: &Path) -> PathBuf {
    path.with_extension("gguf.verified")
}

fn write_verified(path: &Path, expected_size: u64, expected_sha256: &str) -> Result<()> {
    let stamp = file_stamp(path).with_context(|| format!("stat {}", path.display()))?;
    if stamp.size != expected_size {
        return Err(anyhow!(
            "model size changed before verification marker write: {}",
            path.display()
        ));
    }
    let verified = VerifiedFile {
        schema_version: VERIFIED_SCHEMA_VERSION,
        sha256: expected_sha256.to_owned(),
        size: stamp.size,
        modified_ns: stamp.modified_ns,
    };
    let data = serde_json::to_vec(&verified)?;
    let marker = verified_path(path);
    let _ = fs::remove_file(&marker);
    crate::engine::repo::write_atomic(&marker, &data)
}

fn unique_part(target: &Path) -> Result<PathBuf> {
    static NEXT_PART: AtomicU64 = AtomicU64::new(0);
    let sequence = NEXT_PART.fetch_add(1, Ordering::Relaxed);
    let part = target.with_extension(format!("{}.{}.part", std::process::id(), sequence));
    if part.exists() {
        return Err(anyhow!(
            "temporary model file already exists: {}",
            part.display()
        ));
    }
    Ok(part)
}

fn remove_stale_parts(target: &Path) -> Result<()> {
    let parent = target
        .parent()
        .with_context(|| format!("model path has no parent: {}", target.display()))?;
    let stem = target
        .file_stem()
        .with_context(|| format!("model path has no file name: {}", target.display()))?
        .to_string_lossy();
    let prefix = format!("{stem}.");
    for entry in fs::read_dir(parent).with_context(|| format!("read {}", parent.display()))? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(&prefix) && name.ends_with(".part") {
            fs::remove_file(entry.path()).with_context(|| {
                format!("remove stale model download {}", entry.path().display())
            })?;
        }
    }
    Ok(())
}

struct ModelLock {
    _file: fs::File,
}

impl ModelLock {
    fn acquire(path: &Path) -> Result<Self> {
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("open model cache lock {}", path.display()))?;
        let deadline = Instant::now() + LOCK_TIMEOUT;
        loop {
            match file.try_lock() {
                Ok(()) => return Ok(Self { _file: file }),
                Err(fs::TryLockError::WouldBlock) => {
                    if Instant::now() >= deadline {
                        return Err(anyhow!(
                            "timed out waiting for model cache lock {}",
                            path.display()
                        ));
                    }
                    thread::sleep(Duration::from_millis(100));
                }
                Err(fs::TryLockError::Error(error)) => {
                    return Err(error).with_context(|| format!("lock {}", path.display()));
                }
            }
        }
    }
}
