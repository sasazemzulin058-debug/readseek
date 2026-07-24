// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

use anyhow::{Context, Result, anyhow, bail};
use argh::FromArgs;
use rayon::prelude::*;
use std::{env, path::Path, process};
use tree_sitter::Parser;

use crate::cli;
use crate::cli::GitSelection;
use crate::engine::document::NodeKind;
use crate::engine::flags::GitFlags;
use crate::engine::lang::Language;
use crate::engine::output::SearchOutput;
use crate::engine::paths::command_paths;
use crate::engine::source::SourceFile;
use crate::engine::target::Target;
use crate::engine::{def, document_store, document_view, output, refs, rename, repo, vision_cache};

/// Parses arguments and runs the requested command, writing its output.
pub(crate) fn run() -> Result<()> {
    let cli = parse_cli()?;
    if cli.version {
        println!("readseek {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    if let Some(dir) = cli.readseek_dir {
        crate::engine::repo::set_dir_override(dir);
    }
    let output_path = cli.output;
    let command = cli.command.context("command required")?;

    let output = match command {
        cli::Command::Detect(command) => command.run()?,
        cli::Command::Read(command) => command.run()?,
        cli::Command::View(command) => command.run()?,
        cli::Command::Map(command) => command.run()?,
        cli::Command::Check(command) => command.run()?,
        cli::Command::Symbol(command) => command.run()?,
        cli::Command::Identify(command) => command.run()?,
        cli::Command::Def(command) => command.run()?,
        cli::Command::Refs(command) => command.run()?,
        cli::Command::Rename(command) => command.run()?,
        cli::Command::Search(command) => command.run()?,
        cli::Command::Init(command) => {
            command.run()?;
            return Ok(());
        }
    };

    write_output(&output, output_path.as_deref())
}

/// Parse process arguments into a [`cli::Cli`], mirroring `argh::from_env`
/// except that parse failures exit with status 2 (usage error) instead of 1,
/// matching the contract documented in the manual page.
fn parse_cli() -> Result<cli::Cli> {
    let args: Vec<String> = env::args_os()
        .map(std::ffi::OsString::into_string)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|arg| anyhow!("invalid utf-8 argument: {}", arg.to_string_lossy()))?;
    let cmd = args
        .first()
        .and_then(|s| Path::new(s).file_stem().and_then(|n| n.to_str()))
        .unwrap_or("readseek");
    let cli_args: Vec<&str> = args.iter().skip(1).map(String::as_str).collect();
    match cli::Cli::from_args(&[cmd], &cli_args) {
        Ok(cli) => {
            if cli.output.is_some()
                && (cli.version || matches!(&cli.command, Some(cli::Command::Init(_))))
            {
                usage_error(
                    cmd,
                    "--output is only valid with commands that produce JSON",
                );
            }
            if matches!(
                &cli.command,
                Some(cli::Command::Read(command))
                    if command.end.is_some() && command.limit.is_some()
            ) {
                usage_error(cmd, "cannot combine --end with --limit");
            }
            if matches!(
                &cli.command,
                Some(cli::Command::Read(command)) if command.limit == Some(0)
            ) {
                usage_error(cmd, "limit must be greater than zero");
            }
            if matches!(
                &cli.command,
                Some(cli::Command::Read(command)) if command.page == Some(0)
            ) || matches!(
                &cli.command,
                Some(cli::Command::View(command)) if command.page == Some(0)
            ) {
                usage_error(cmd, "page must be greater than zero");
            }
            if matches!(
                &cli.command,
                Some(cli::Command::Read(command))
                    if command.vision_mode.is_some()
                        && (command.end.is_some()
                            || command.limit.is_some()
                            || command.language.is_some())
            ) {
                usage_error(
                    cmd,
                    "--vision-mode cannot be combined with --end, --limit, or --language",
                );
            }
            if matches!(
                &cli.command,
                Some(cli::Command::Read(command))
                    if command.vision_level.is_some()
                        && (command.end.is_some()
                            || command.limit.is_some()
                            || command.language.is_some())
            ) {
                usage_error(
                    cmd,
                    "--vision-level cannot be combined with --end, --limit, or --language",
                );
            }
            if let Some(cli::Command::View(view)) = &cli.command
                && let Some(kind) = &view.kind
                && let Err(error) = NodeKind::parse(kind)
            {
                usage_error(cmd, &error.to_string());
            }
            Ok(cli)
        }
        Err(early_exit) if early_exit.status.is_ok() => {
            println!("{}", early_exit.output);
            process::exit(0);
        }
        Err(early_exit) => usage_error(cmd, &early_exit.output),
    }
}

fn usage_error(cmd: &str, message: &str) -> ! {
    eprintln!("{message}\nRun {cmd} --help for more information.");
    process::exit(2);
}

impl cli::DetectCommand {
    fn run(&self) -> Result<String> {
        let source = load_path_source(self.target.as_deref(), None)?;
        let output = output::DetectOutput::from_detection(source.detection);
        Ok(serde_json::to_string(&output)?)
    }
}

/// Run the requested vision tasks, reusing cached results from `readseek_dir`.
fn run_vision(
    input: crate::engine::qwen::VisionInput<'_>,
    request: crate::engine::vision::Request,
    level: crate::engine::vision::VisionLevel,
    readseek_dir: Option<&Path>,
) -> Result<crate::engine::vision::Analysis> {
    let hash = input.cache_hash();

    let (mut entry, cache_version) = readseek_dir.map_or_else(
        || {
            (
                vision_cache::CacheEntry::new_empty(level),
                vision_cache::CacheVersion::Missing,
            )
        },
        |dir| vision_cache::load(dir, &hash, level),
    );
    let requested_tasks = [request.caption, request.objects, request.ocr]
        .into_iter()
        .filter(|requested| *requested)
        .count();
    let cached_tasks = [
        request.caption && entry.caption.is_some(),
        request.objects && entry.objects.is_some(),
        request.ocr && entry.ocr.is_some(),
    ]
    .into_iter()
    .filter(|cached| *cached)
    .count();
    let cache_status = if cached_tasks == requested_tasks {
        "hit"
    } else if cached_tasks == 0 {
        "miss"
    } else {
        "partial"
    };
    tracing::trace!(
        target: "tracing",
        cache = cache_status,
        vision_level = ?level,
        requested_tasks,
        cached_tasks,
        "vision cache"
    );

    let missing = crate::engine::vision::Request {
        caption: request.caption && entry.caption.is_none(),
        objects: request.objects && entry.objects.is_none(),
        ocr: request.ocr && entry.ocr.is_none(),
    };
    if missing.caption || missing.objects || missing.ocr {
        let analysis = crate::engine::vision::analyze(input, missing, level)?;
        if missing.caption {
            entry.caption = analysis.caption;
        }
        if missing.objects {
            entry.objects = analysis.objects;
        }
        if missing.ocr {
            entry.ocr = analysis.ocr;
        }
        if let Some(dir) = readseek_dir {
            vision_cache::store(dir, &hash, &cache_version, &entry);
        }
    }

    Ok(crate::engine::vision::Analysis {
        caption: if request.caption { entry.caption } else { None },
        objects: if request.objects { entry.objects } else { None },
        ocr: if request.ocr { entry.ocr } else { None },
    })
}

fn run_pdf_vision(
    input: crate::engine::qwen::VisionInput<'_>,
    request: crate::engine::vision::Request,
    level: crate::engine::vision::VisionLevel,
    readseek_dir: Option<&Path>,
) -> crate::engine::vision::Analysis {
    match run_vision(input, request, level, readseek_dir) {
        Ok(analysis) => analysis,
        Err(error) => {
            tracing::warn!("vision skipped: {error:#}");
            crate::engine::vision::Analysis::default()
        }
    }
}

impl cli::ReadCommand {
    fn run(&self) -> Result<String> {
        let (target, source) = load_source(self.target.as_deref(), false, self.language)?;
        let readseek_dir = repo::find_readseek_dir(&source.path).or_else(|| {
            env::current_dir()
                .ok()
                .and_then(|cwd| repo::find_readseek_dir(&cwd))
        });
        let vision_options = self.vision_level.is_some();
        let level = self.vision_level.unwrap_or_default();
        if matches!(
            source.detection.category,
            crate::engine::source::ContentCategory::Image(_)
        ) {
            if target.address.is_some() {
                bail!("image targets do not support a line or hash suffix");
            }
            if self.end.is_some()
                || self.limit.is_some()
                || self.page.is_some()
                || self.language.is_some()
            {
                bail!("--end, --limit, --page, and --language do not apply to images");
            }
            let mode = self.vision_mode.unwrap_or_default();
            if mode == cli::ImageMode::None && vision_options {
                bail!("--vision-level requires an analysis --vision-mode");
            }
            let Some(bytes) = source.document_bytes.as_deref() else {
                bail!("missing image bytes for {}", source.path.display());
            };
            let request = crate::engine::vision::Request {
                caption: mode.includes_caption(),
                objects: mode.includes_objects(),
                ocr: mode.includes_ocr(),
            };
            let analysis = run_vision(
                crate::engine::qwen::VisionInput::Encoded(bytes),
                request,
                level,
                readseek_dir.as_deref(),
            )?;
            let prepared = (mode == cli::ImageMode::None)
                .then(|| crate::engine::image::preprocess(bytes))
                .transpose()?;
            let image_output = output::read_image_output(&source, mode, analysis, prepared)?;
            let output = output::ReadOutput::Image(image_output);
            return Ok(serde_json::to_string(&output)?);
        }

        if matches!(
            source.detection.category,
            crate::engine::source::ContentCategory::Pdf(_)
        ) {
            if target.address.is_some() {
                bail!("PDF targets do not support a line or hash suffix");
            }
            if self.end.is_some() || self.limit.is_some() || self.language.is_some() {
                bail!("--end, --limit, and --language do not apply to PDFs");
            }
            let mode = self.vision_mode.unwrap_or_default();
            if mode == cli::ImageMode::None && vision_options {
                bail!("--vision-level requires an analysis --vision-mode");
            }
            let Some(bytes) = source.document_bytes.as_deref() else {
                bail!("missing PDF bytes for {}", source.path.display());
            };
            let pdf = crate::engine::pdf::read(bytes, mode, self.page, |input, request| {
                run_pdf_vision(input, request, level, readseek_dir.as_deref())
            })?;
            return Ok(serde_json::to_string(&output::ReadOutput::Pdf(pdf))?);
        }

        if self.page.is_some() {
            bail!("--page applies to PDFs only");
        }
        if vision_options {
            bail!("vision options apply to image and PDF files only");
        }

        source.require_text()?;
        let start = output::resolve_target(&source, &target)?;

        let end = if let Some(limit) = self.limit {
            let start_line = start.unwrap_or(1);
            Some(
                start_line
                    .checked_add(limit - 1)
                    .context("read range exceeds supported line numbers")?,
            )
        } else {
            self.end
        };
        let text_output = output::read_output(&source, start, end)?;
        let output = output::ReadOutput::Text(text_output);
        Ok(serde_json::to_string(&output)?)
    }
}

impl cli::ViewCommand {
    fn run(&self) -> Result<String> {
        let source = load_path_source(self.target.as_deref(), None)?;
        if !matches!(
            source.detection.category,
            crate::engine::source::ContentCategory::Pdf(_)
        ) {
            bail!("view currently supports PDF documents only");
        }
        let bytes = source
            .document_bytes
            .as_deref()
            .context("missing PDF document bytes")?;
        let id = crate::engine::hash::hash_bytes(bytes);
        let readseek_dir = repo::find_readseek_dir(&source.path)
            .or_else(|| {
                env::current_dir()
                    .ok()
                    .and_then(|cwd| repo::find_readseek_dir(&cwd))
            })
            .context("no .readseek directory found; run 'readseek init' first")?;
        let mut document = if let Some(document) = document_store::load(&readseek_dir, &id)? {
            document
        } else if self.outline {
            crate::engine::pdf::extract_outline_document(&source.path, bytes, id)?
        } else {
            let assets_dir = document_store::assets_dir(&readseek_dir, &id);
            let document =
                crate::engine::pdf::extract_document(&source.path, bytes, id, &assets_dir)?;
            document_store::store(&readseek_dir, &document)?;
            document
        };
        document.rebind_source(&source.path);
        if let Some(page) = self.page
            && page > document.pages
        {
            bail!("page {page} exceeds document page count {}", document.pages);
        }
        let selection = document_view::Selection {
            node: self.node.as_deref(),
            page: self.page,
            kind: self.kind.as_deref().map(NodeKind::parse).transpose()?,
            depth: self.depth,
            outline: self.outline,
            overview: self.node.is_none()
                && self.page.is_none()
                && self.kind.is_none()
                && !self.outline,
        };
        let document = document_view::select(&document, selection)?;

        match self.format {
            output::Format::Plain => Ok(document_view::render(&document)),
            output::Format::Json => Ok(serde_json::to_string(&document)?),
        }
    }
}

impl cli::MapCommand {
    fn run(&self) -> Result<String> {
        let source = load_path_source(self.target.as_deref(), self.language)?;
        source.require_text()?;
        Ok(serde_json::to_string(&output::map_output(&source)?)?)
    }
}

impl cli::CheckCommand {
    fn run(&self) -> Result<String> {
        let source = load_path_source(self.target.as_deref(), self.language)?;
        source.require_text()?;
        Ok(serde_json::to_string(&output::check_output(&source)?)?)
    }
}

impl cli::SymbolCommand {
    fn run(&self) -> Result<String> {
        let (target, source) = load_source(self.target.as_deref(), self.name, self.language)?;
        source.require_text()?;
        let address = if let Some(crate::engine::target::TargetAddress::Name(name)) =
            target.address.as_ref()
        {
            output::SymbolAddress::Name(name)
        } else {
            if self.name {
                bail!("--name requires a target name suffix; use <target>:<name> --name");
            }
            let target_line = output::resolve_target(&source, &target)?;
            match target_line {
                Some(line) => output::SymbolAddress::Line(line),
                None => bail!("symbol requires a name or target line/hash"),
            }
        };
        let output = output::symbol_output(&source, address)?;
        Ok(serde_json::to_string(&output)?)
    }
}

impl cli::IdentifyCommand {
    fn run(&self) -> Result<String> {
        let (target, source) = load_source(self.target.as_deref(), false, self.language)?;
        source.require_text()?;
        let target_line = output::resolve_target(&source, &target)?;
        let output = output::identify_output(&source, target_line, self.column)?;
        Ok(serde_json::to_string(&output)?)
    }
}

impl cli::DefCommand {
    fn run(self) -> Result<String> {
        let flags = self.git_flags();
        let request = def::Request {
            target: self.target,
            name: self.name,
            language: self.language,
            flags,
        };
        let output = def::output(&request)?;
        match self.format {
            output::Format::Plain => Ok(serde_json::to_string(&def::compact(&output))?),
            output::Format::Json => Ok(serde_json::to_string(&output)?),
        }
    }
}

impl cli::RefsCommand {
    fn run(self) -> Result<String> {
        let flags = self.git_flags();
        let request = refs::Request {
            target: self.target,
            name: self.name,
            scope: self.scope,
            line: self.line,
            column: self.column,
            language: self.language,
            flags,
        };
        let output = refs::output(&request)?;
        match self.format {
            output::Format::Plain => Ok(serde_json::to_string(&refs::compact(&output))?),
            output::Format::Json => Ok(serde_json::to_string(&output)?),
        }
    }
}

impl cli::RenameCommand {
    fn run(self) -> Result<String> {
        let flags = self.git_flags();
        let request = rename::Request {
            target: self.target,
            line: self.line,
            column: self.column,
            to: self.to,
            workspace: self.workspace,
            apply: self.apply,
            expected_plan_hash: self.plan_hash,
            language: self.language,
            flags,
        };
        Ok(serde_json::to_string(&rename::output(&request)?)?)
    }
}

impl cli::SearchCommand {
    fn run(&self) -> Result<String> {
        let paths = command_paths(&self.target, self.git_flags())?;
        let mut pattern = crate::engine::search::compile_search(&self.pattern);
        if let Some(language) = self
            .language
            .and_then(crate::engine::symbols::tree_sitter_language)
        {
            crate::engine::search::prepare_tree(&mut pattern, &language);
        }

        let results: Vec<_> = paths
            .par_iter()
            .map_init(Parser::new, |parser, path| {
                crate::engine::search::search_file(path, self.language, &pattern, parser)
                    .map(|result| result.filter(|result| !result.matches.is_empty()))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();

        Ok(serde_json::to_string(&SearchOutput { results })?)
    }
}

impl cli::InitCommand {
    fn run(&self) -> Result<()> {
        let path = self.path.as_deref().unwrap_or(Path::new("."));
        let init = repo::init(path)?;
        repo::update(
            path,
            GitFlags {
                cached: true,
                others: true,
                ignored: false,
            },
        )?;
        if init.reinitialized {
            println!(
                "Reinitialized existing readseek repository in {}/",
                init.path.display()
            );
        } else {
            println!(
                "Initialized empty readseek repository in {}/",
                init.path.display()
            );
        }
        Ok(())
    }
}

fn load_source(
    target_str: Option<&str>,
    name_mode: bool,
    language: Option<Language>,
) -> Result<(Target, SourceFile)> {
    let target = crate::cli::parse_target(target_str.context("target required")?, name_mode)?;
    let source = output::load_source_for_input(&target, language)?;
    Ok((target, source))
}

fn load_path_source(target_str: Option<&str>, language: Option<Language>) -> Result<SourceFile> {
    let target = crate::cli::parse_target(target_str.context("target required")?, false)?;
    if target.address.is_some() {
        bail!("this command takes a file path, not a line or hash suffix");
    }
    output::load_source_for_input(&target, language)
}

fn write_output(output: &str, path: Option<&Path>) -> Result<()> {
    if let Some(path) = path {
        std::fs::write(path, output).with_context(|| format!("write {}", path.display()))
    } else {
        println!("{output}");
        Ok(())
    }
}
