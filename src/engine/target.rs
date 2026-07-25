// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

use anyhow::{Context, Result, bail};
use std::path::PathBuf;

use crate::engine::hash::LineHash;
use crate::engine::source::SourceFile;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Target {
    pub(crate) path: PathBuf,
    pub(crate) address: Option<TargetAddress>,
    pub(crate) read_stdin: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TargetAddress {
    Line(usize),
    Hash(String),
    Name(String),
}

impl Target {
    pub(crate) fn parse(value: &str, name_mode: bool) -> Result<Self> {
        if value.is_empty() {
            bail!("target must not be empty");
        }

        let (read_stdin, rest) = match value.strip_prefix("stdin:") {
            Some(rest) => (true, rest),
            None => (false, value),
        };

        let (path, address) = if rest.is_empty() {
            (PathBuf::new(), None)
        } else if let Some((path, suffix)) = rest.rsplit_once(':') {
            let address = if suffix.is_empty() {
                None
            } else if name_mode {
                Some(TargetAddress::Name(suffix.to_owned()))
            } else if suffix.chars().all(|ch| ch.is_ascii_digit()) {
                let line = suffix
                    .parse::<usize>()
                    .with_context(|| format!("invalid target line: {suffix}"))?;
                if line == 0 {
                    bail!("target line must be greater than zero");
                }
                Some(TargetAddress::Line(line))
            } else if LineHash::is_valid_str(suffix) {
                Some(TargetAddress::Hash(suffix.to_ascii_lowercase()))
            } else {
                None
            };
            if address.is_some() {
                (PathBuf::from(path), address)
            } else {
                (PathBuf::from(rest), None)
            }
        } else {
            (PathBuf::from(rest), None)
        };

        if !read_stdin && path.as_os_str().is_empty() {
            bail!("target path must not be empty");
        }

        Ok(Self {
            path,
            address,
            read_stdin,
        })
    }

    pub(crate) fn resolve_line(&self, source: &SourceFile) -> Result<Option<usize>> {
        self.address
            .as_ref()
            .map(|addr| addr.resolve(source))
            .transpose()
    }
}

impl TargetAddress {
    pub(crate) fn resolve(&self, source: &SourceFile) -> Result<usize> {
        match self {
            Self::Line(line) => Ok(*line),
            Self::Hash(hash) => {
                let target = hash
                    .parse::<LineHash>()
                    .with_context(|| format!("invalid target hash {hash}"))?;
                source
                    .lines
                    .iter()
                    .find_map(|line| (line.hash() == target).then_some(line.number))
                    .with_context(|| format!("hash {hash} not found in {}", source.path.display()))
            }
            Self::Name(_) => bail!("name targets are not supported for this command"),
        }
    }

    pub(crate) fn name(&self) -> Option<&str> {
        match self {
            Self::Name(name) => Some(name),
            _ => None,
        }
    }
}
