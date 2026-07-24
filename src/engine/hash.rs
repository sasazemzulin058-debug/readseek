// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

use std::fmt;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use serde::{Serialize, Serializer};

pub(crate) const HASHLINE_MODULUS: u32 = 0x1000;

/// A locality-preserving line hash: a value in `[0, HASHLINE_MODULUS)` rendered
/// as a three-digit hex string.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct LineHash(u16);

impl LineHash {
    /// Wrap a raw value, rejecting anything at or above [`HASHLINE_MODULUS`].
    pub(crate) fn new(value: u16) -> Result<Self> {
        Self::try_from(value)
    }

    /// The raw value, always in `[0, HASHLINE_MODULUS)`.
    #[allow(dead_code)]
    pub(crate) fn as_u16(self) -> u16 {
        u16::from(self)
    }
}

impl From<LineHash> for u16 {
    fn from(hash: LineHash) -> Self {
        hash.0
    }
}

impl TryFrom<u16> for LineHash {
    type Error = anyhow::Error;

    fn try_from(value: u16) -> Result<Self> {
        if u32::from(value) >= HASHLINE_MODULUS {
            bail!("line hash {value:#x} is out of range");
        }
        Ok(Self(value))
    }
}

impl fmt::Display for LineHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:03x}", self.0)
    }
}

impl FromStr for LineHash {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let raw = u16::from_str_radix(value, 16)
            .with_context(|| format!("invalid line hash `{value}`"))?;
        Self::new(raw)
    }
}

impl Serialize for LineHash {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

/// Compute a stable locality-preserving hash for a source line.
///
/// The line text is whitespace-normalized before hashing so that minor
/// whitespace changes do not invalidate the hash.
pub(crate) fn hash_line(text: &str) -> LineHash {
    let text = text.strip_suffix('\r').unwrap_or(text);
    let mut hasher = xxhash_rust::xxh32::Xxh32::new(0);
    for token in text.split_whitespace() {
        hasher.update(token.as_bytes());
    }
    let digest = hasher.digest() % HASHLINE_MODULUS;
    LineHash(u16::try_from(digest).unwrap_or_default())
}

/// Compute a BLAKE3 content hash for the full source text.
pub(crate) fn hash_text(text: &str) -> String {
    blake3::hash(text.as_bytes()).to_string()
}

/// Compute a BLAKE3 content hash for raw bytes as a 64-char lowercase hex string.
pub(crate) fn hash_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_string()
}

/// Check if a string is a valid hex string of expected length (or any length if expected is None).
pub(crate) fn is_hex(s: &str, expected_len: Option<usize>) -> bool {
    if expected_len.is_some_and(|len| s.len() != len) {
        return false;
    }
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Decode a hex string into a fixed-size byte array.
pub(crate) fn decode_hex<const N: usize>(s: &str) -> Result<[u8; N]> {
    if s.len() != N * 2 {
        bail!("invalid hex length {}, expected {}", s.len(), N * 2);
    }
    let mut out = [0u8; N];
    for (i, chunk) in s.as_bytes().chunks_exact(2).enumerate() {
        let h = match chunk[0] {
            b'0'..=b'9' => chunk[0] - b'0',
            b'a'..=b'f' => chunk[0] - b'a' + 10,
            b'A'..=b'F' => chunk[0] - b'A' + 10,
            _ => bail!("invalid hex character"),
        };
        let l = match chunk[1] {
            b'0'..=b'9' => chunk[1] - b'0',
            b'a'..=b'f' => chunk[1] - b'a' + 10,
            b'A'..=b'F' => chunk[1] - b'A' + 10,
            _ => bail!("invalid hex character"),
        };
        out[i] = (h << 4) | l;
    }
    Ok(out)
}
