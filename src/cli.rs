// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

use crate::engine::flags::GitFlags;
use crate::engine::lang::{LANGUAGE_SPECS, Language};
use crate::engine::output::ImageMode;
use crate::engine::target::Target;
use crate::engine::vision::VisionLevel;
use anyhow::Result;
use argh::FromArgs;
use std::path::PathBuf;

pub(crate) mod run;

/// readseek
#[derive(Debug, FromArgs)]
#[argh(help_triggers("-h", "--help"))]
pub(crate) struct Cli {
    /// print version and exit
    #[argh(switch, short = 'V')]
    pub(crate) version: bool,

    /// write output to file instead of stdout
    #[argh(option, long = "output")]
    pub(crate) output: Option<PathBuf>,

    /// use the given .readseek directory instead of discovering one
    #[argh(option, long = "readseek-dir")]
    pub(crate) readseek_dir: Option<PathBuf>,

    /// command to run
    #[argh(subcommand)]
    pub(crate) command: Option<Command>,
}

#[derive(Debug, FromArgs)]
#[argh(subcommand)]
pub(crate) enum Command {
    Detect(DetectCommand),
    Read(ReadCommand),
    View(ViewCommand),
    Map(MapCommand),
    Check(CheckCommand),
    Symbol(SymbolCommand),
    Identify(IdentifyCommand),
    Def(DefCommand),
    Refs(RefsCommand),
    Rename(RenameCommand),
    Search(SearchCommand),
    Init(InitCommand),
}

/// detect the file type
#[derive(Debug, FromArgs)]
#[argh(subcommand, name = "detect")]
#[argh(help_triggers("-h", "--help"))]
pub(crate) struct DetectCommand {
    /// takes <file> or stdin:<path>
    #[argh(positional)]
    pub(crate) target: Option<String>,
}

/// read and hash from a line range
#[derive(Debug, FromArgs)]
#[argh(subcommand, name = "read")]
#[argh(help_triggers("-h", "--help"))]
pub(crate) struct ReadCommand {
    /// takes <file>, <file>:<line>, <file>:<hash>, stdin:<path>[:<line>|<hash>], or stdin:
    #[argh(positional)]
    pub(crate) target: Option<String>,

    /// last line to include
    #[argh(option)]
    pub(crate) end: Option<usize>,

    /// maximum number of lines to include
    #[argh(option)]
    pub(crate) limit: Option<usize>,

    /// one-based PDF page to read
    #[argh(option)]
    pub(crate) page: Option<usize>,

    /// language override
    #[argh(option, from_str_fn(parse_language))]
    pub(crate) language: Option<Language>,

    /// vision mode: none (default), all, caption, objects, or ocr
    #[argh(option)]
    pub(crate) vision_mode: Option<ImageMode>,

    /// vision inference level: low (default), medium, or high
    #[argh(option)]
    pub(crate) vision_level: Option<VisionLevel>,
}

/// view an indexed document
#[derive(Debug, FromArgs)]
#[argh(subcommand, name = "view")]
#[argh(help_triggers("-h", "--help"))]
pub(crate) struct ViewCommand {
    /// document file to view
    #[argh(positional)]
    pub(crate) target: Option<String>,

    /// output format: plain (default) or json
    #[argh(
        option,
        long = "format",
        default = "crate::engine::output::Format::Plain"
    )]
    pub(crate) format: crate::engine::output::Format,

    /// node ID to use as the view root
    #[argh(option)]
    pub(crate) node: Option<String>,

    /// one-based source page to view
    #[argh(option)]
    pub(crate) page: Option<usize>,

    /// node kind to include
    #[argh(option)]
    pub(crate) kind: Option<String>,

    /// maximum node depth to include
    #[argh(option)]
    pub(crate) depth: Option<usize>,

    /// show the document outline only
    #[argh(switch)]
    pub(crate) outline: bool,
}

/// map a file to symbols
#[derive(Debug, FromArgs)]
#[argh(subcommand, name = "map")]
#[argh(help_triggers("-h", "--help"))]
pub(crate) struct MapCommand {
    /// takes <file> or stdin:<path>
    #[argh(positional)]
    pub(crate) target: Option<String>,

    /// language override
    #[argh(option, from_str_fn(parse_language))]
    pub(crate) language: Option<Language>,
}

/// report parse diagnostics for a file
#[derive(Debug, FromArgs)]
#[argh(subcommand, name = "check")]
#[argh(help_triggers("-h", "--help"))]
pub(crate) struct CheckCommand {
    /// takes <file> or stdin:<path>
    #[argh(positional)]
    pub(crate) target: Option<String>,

    /// language override
    #[argh(option, from_str_fn(parse_language))]
    pub(crate) language: Option<Language>,
}

/// read the line range for a symbol
#[derive(Debug, FromArgs)]
#[argh(subcommand, name = "symbol")]
#[argh(help_triggers("-h", "--help"))]
pub(crate) struct SymbolCommand {
    /// takes <file>, <file>:<line>, <file>:<hash>, stdin:<path>[:<line>|<hash>], or stdin:
    #[argh(positional)]
    pub(crate) target: Option<String>,

    /// treat the target suffix as a symbol name
    #[argh(switch)]
    pub(crate) name: bool,

    /// language override
    #[argh(option, from_str_fn(parse_language))]
    pub(crate) language: Option<Language>,
}

/// identify the cursor token and enclosing symbol
#[derive(Debug, FromArgs)]
#[argh(subcommand, name = "identify")]
#[argh(help_triggers("-h", "--help"))]
pub(crate) struct IdentifyCommand {
    /// takes <file>, <file>:<line>, <file>:<hash>, stdin:<path>[:<line>|<hash>], or stdin:
    #[argh(positional)]
    pub(crate) target: Option<String>,

    /// one-based cursor byte column
    #[argh(option)]
    pub(crate) column: Option<usize>,

    /// language override
    #[argh(option, from_str_fn(parse_language))]
    pub(crate) language: Option<Language>,
}

/// find structural symbol definitions
#[derive(Debug, FromArgs)]
#[argh(subcommand, name = "def")]
#[argh(help_triggers("-h", "--help"))]
pub(crate) struct DefCommand {
    /// file or directory to search
    #[argh(positional)]
    pub(crate) target: PathBuf,

    /// qualified symbol name or unqualified name
    #[argh(positional)]
    pub(crate) name: String,

    /// output format
    #[argh(
        option,
        long = "format",
        default = "crate::engine::output::Format::Json"
    )]
    pub(crate) format: crate::engine::output::Format,

    /// language override
    #[argh(option, from_str_fn(parse_language))]
    pub(crate) language: Option<Language>,

    /// comma-list of Git search scopes: cached, others, ignored (default none)
    #[argh(option, short = 'g', default = "GitGroup::EMPTY")]
    pub(crate) git: GitGroup,
}

/// find identifier references
#[derive(Debug, FromArgs)]
#[argh(subcommand, name = "refs")]
#[argh(help_triggers("-h", "--help"))]
pub(crate) struct RefsCommand {
    /// file or directory to search
    #[argh(positional)]
    pub(crate) target: PathBuf,

    /// identifier to search for
    #[argh(positional)]
    pub(crate) name: String,

    /// output format
    #[argh(
        option,
        long = "format",
        default = "crate::engine::output::Format::Json"
    )]
    pub(crate) format: crate::engine::output::Format,

    /// restrict results to the binding under --line/--column (single file)
    #[argh(switch)]
    pub(crate) scope: bool,

    /// one-based cursor line, used with --scope
    #[argh(option)]
    pub(crate) line: Option<usize>,

    /// one-based cursor byte column, used with --scope
    #[argh(option)]
    pub(crate) column: Option<usize>,

    /// language override
    #[argh(option, from_str_fn(parse_language))]
    pub(crate) language: Option<Language>,

    /// comma-list of Git search scopes: cached, others, ignored (default none)
    #[argh(option, short = 'g', default = "GitGroup::EMPTY")]
    pub(crate) git: GitGroup,
}

/// plan a binding-accurate rename within a single file
#[derive(Debug, FromArgs)]
#[argh(subcommand, name = "rename")]
#[argh(help_triggers("-h", "--help"))]
pub(crate) struct RenameCommand {
    /// file holding the binding under the cursor (a single regular file)
    #[argh(positional)]
    pub(crate) target: PathBuf,

    /// one-based cursor line of the binding to rename
    #[argh(option)]
    pub(crate) line: usize,

    /// one-based cursor byte column of the binding to rename
    #[argh(option)]
    pub(crate) column: Option<usize>,

    /// new name for the binding
    #[argh(option, long = "to")]
    pub(crate) to: String,

    /// expand the rename across this directory or repository (name-based
    /// outside the cursor file); omit for a single-file rename
    #[argh(option)]
    pub(crate) workspace: Option<PathBuf>,

    /// write the planned edits to the file after verifying line hashes
    #[argh(switch)]
    pub(crate) apply: bool,

    /// require the apply plan to match this dry-run plan hash
    #[argh(option)]
    pub(crate) plan_hash: Option<String>,

    /// language override
    #[argh(option, from_str_fn(parse_language))]
    pub(crate) language: Option<Language>,

    /// comma-list of Git search scopes when expanding across a repository:
    /// cached, others, ignored (default none)
    #[argh(option, short = 'g', default = "GitGroup::EMPTY")]
    pub(crate) git: GitGroup,
}

/// search files with an AST pattern
#[derive(Debug, FromArgs)]
#[argh(subcommand, name = "search")]
#[argh(help_triggers("-h", "--help"))]
pub(crate) struct SearchCommand {
    /// file or directory to search
    #[argh(positional)]
    pub(crate) target: PathBuf,

    /// ast-grep-style pattern
    #[argh(positional)]
    pub(crate) pattern: String,

    /// language override
    #[argh(option, from_str_fn(parse_language))]
    pub(crate) language: Option<Language>,

    /// comma-list of Git search scopes: cached, others, ignored (default none)
    #[argh(option, short = 'g', default = "GitGroup::EMPTY")]
    pub(crate) git: GitGroup,
}

/// initialize .readseek/ directory
#[derive(Debug, FromArgs)]
#[argh(subcommand, name = "init")]
#[argh(help_triggers("-h", "--help"))]
pub(crate) struct InitCommand {
    /// path to a directory (defaults to current directory)
    #[argh(positional)]
    pub(crate) path: Option<PathBuf>,
}

pub(crate) fn parse_language(value: &str) -> std::result::Result<Language, String> {
    if let Ok(language) = value.parse::<Language>() {
        return Ok(language);
    }

    let alias = value.to_ascii_lowercase().replace(['-', '_'], "");
    LANGUAGE_SPECS
        .iter()
        .find_map(|spec| {
            spec.aliases
                .contains(&alias.as_str())
                .then_some(spec.language)
        })
        .ok_or_else(|| format!("unknown language `{value}`"))
}
pub(crate) fn parse_target(value: &str, name_mode: bool) -> Result<Target> {
    Target::parse(value, name_mode)
}

/// Comma-list of Git search scopes requested via `--git`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GitGroup(u8);

impl GitGroup {
    const CACHED: u8 = 1;
    const OTHERS: u8 = 2;
    const IGNORED: u8 = 4;
    pub(crate) const EMPTY: Self = Self(0);

    fn contains(self, bit: u8) -> bool {
        self.0 & bit != 0
    }

    pub(crate) fn to_flags(self) -> GitFlags {
        GitFlags {
            cached: self.contains(Self::CACHED),
            others: self.contains(Self::OTHERS),
            ignored: self.contains(Self::IGNORED),
        }
    }
}

impl argh::FromArgValue for GitGroup {
    fn from_arg_value(value: &str) -> std::result::Result<Self, String> {
        let mut bits = 0_u8;
        for token in value.split(',') {
            bits |= match token.trim() {
                "cached" => Self::CACHED,
                "others" => Self::OTHERS,
                "ignored" => Self::IGNORED,
                other => {
                    return Err(format!(
                        "unknown git scope `{other}`; expected cached, others, or ignored"
                    ));
                }
            };
        }
        Ok(Self(bits))
    }
}

pub(crate) trait GitSelection {
    fn git_flags(&self) -> GitFlags;
}

macro_rules! impl_git_selection {
    ($($command:ty),+ $(,)?) => {
        $(
            impl GitSelection for $command {
                fn git_flags(&self) -> GitFlags {
                    self.git.to_flags()
                }
            }
        )+
    };
}

impl_git_selection!(DefCommand, RefsCommand, RenameCommand, SearchCommand);
