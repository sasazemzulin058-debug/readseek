// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

use anyhow::Result;
use serde::{Serialize, Serializer};
use std::path::Path;
use std::sync::OnceLock;
use strum_macros::{Display, EnumString, FromRepr};
use syntect::parsing::{SyntaxReference, SyntaxSet};

#[repr(u16)]
#[derive(Clone, Copy, Debug, Display, EnumString, Eq, FromRepr, PartialEq)]
#[strum(serialize_all = "kebab-case", ascii_case_insensitive)]
pub(crate) enum Language {
    Assembly = 0,
    C = 1,
    Bash = 2,
    Cpp = 3,
    CSharp = 4,
    Css = 5,
    Dockerfile = 6,
    Go = 7,
    Gdscript = 8,
    Java = 9,
    JavaScript = 10,
    Jsx = 11,
    Html = 12,
    Json = 13,
    Kconfig = 14,
    Latex = 15,
    Lua = 16,
    Markdown = 17,
    Xml = 18,
    Yaml = 19,
    Just = 20,
    Make = 21,
    Meson = 22,
    Nix = 23,
    Perl = 24,
    Python = 25,
    Php = 26,
    Puppet = 27,
    Ruby = 28,
    Riscv = 29,
    Rust = 30,
    Swift = 31,
    Sql = 32,
    TypeScript = 33,
    Typst = 34,
    Toml = 35,
    Tsx = 36,
    Vimscript = 37,
    Zig = 38,
    Unknown = 39,
}

impl From<Language> for u16 {
    fn from(language: Language) -> Self {
        language as u16
    }
}

static SPEC_BY_LANG: OnceLock<[Option<&'static LanguageSpec>; 40]> = OnceLock::new();

fn init_spec_table() -> [Option<&'static LanguageSpec>; 40] {
    let mut table = [None; 40];
    for spec in LANGUAGE_SPECS {
        table[spec.language as usize] = Some(spec);
    }
    table
}

impl Language {
    pub(crate) fn spec(self) -> Option<&'static LanguageSpec> {
        let table = SPEC_BY_LANG.get_or_init(init_spec_table);
        table.get(self as usize).copied().flatten()
    }

    pub(crate) fn id(self) -> &'static str {
        self.spec().map_or("unknown", |spec| spec.id)
    }

    pub(crate) fn engine(self) -> Option<AnalysisEngine> {
        self.spec().and_then(|spec| spec.engine)
    }

    pub(crate) fn has_symbols(self) -> bool {
        self.spec().is_some_and(|spec| spec.has_symbols)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct LanguageSpec {
    pub(crate) language: Language,
    pub(crate) id: &'static str,
    pub(crate) engine: Option<AnalysisEngine>,
    pub(crate) aliases: &'static [&'static str],
    pub(crate) extensions: &'static [&'static str],
    pub(crate) file_names: &'static [&'static str],
    pub(crate) syntax_names: &'static [&'static str],
    pub(crate) has_symbols: bool,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, FromRepr, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
#[allow(dead_code)]
pub(crate) enum AnalysisEngine {
    TreeSitter = 0,
    Llvm = 1,
}

impl From<AnalysisEngine> for u8 {
    fn from(engine: AnalysisEngine) -> Self {
        engine as u8
    }
}
#[derive(Debug)]
pub(crate) struct EngineField(pub(crate) Option<AnalysisEngine>);

impl EngineField {
    pub(crate) fn is_none(&self) -> bool {
        self.0.is_none()
    }
}

impl Serialize for EngineField {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            Some(engine) => engine.serialize(serializer),
            None => serializer.serialize_str("none"),
        }
    }
}

pub(crate) const LANGUAGE_SPECS: &[LanguageSpec] = &[
    LanguageSpec {
        language: Language::Assembly,
        id: "assembly",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["assembly", "asm", "x86", "arm"],
        extensions: &["asm", "s", "S"],
        file_names: &[],
        syntax_names: &["Assembly", "ARM Assembly", "x86 Assembly"],
        has_symbols: false,
    },
    LanguageSpec {
        language: Language::C,
        id: "c",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["c"],
        extensions: &["c", "h"],
        file_names: &[],
        syntax_names: &["C"],
        has_symbols: true,
    },
    LanguageSpec {
        language: Language::Bash,
        id: "bash",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["bash", "sh", "shell"],
        extensions: &["bash", "sh"],
        file_names: &[".bashrc", ".bash_profile", ".profile"],
        syntax_names: &[
            "Bourne Again Shell (bash)",
            "Shell-Unix-Generic",
            "ShellScript",
        ],
        has_symbols: true,
    },
    LanguageSpec {
        language: Language::Cpp,
        id: "cpp",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["cpp", "cxx", "cplusplus"],
        extensions: &["cc", "cpp", "cxx", "hh", "hpp", "hxx"],
        file_names: &[],
        syntax_names: &["C++"],
        has_symbols: true,
    },
    LanguageSpec {
        language: Language::CSharp,
        id: "csharp",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["csharp", "cs", "c#"],
        extensions: &["cs"],
        file_names: &[],
        syntax_names: &["C#"],
        has_symbols: true,
    },
    LanguageSpec {
        language: Language::Css,
        id: "css",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["css"],
        extensions: &["css"],
        file_names: &[],
        syntax_names: &["CSS"],
        has_symbols: false,
    },
    LanguageSpec {
        language: Language::Dockerfile,
        id: "dockerfile",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["dockerfile", "containerfile"],
        extensions: &[],
        file_names: &["Dockerfile", "Containerfile"],
        syntax_names: &["Dockerfile"],
        has_symbols: false,
    },
    LanguageSpec {
        language: Language::Go,
        id: "go",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["go", "golang"],
        extensions: &["go"],
        file_names: &["go.mod"],
        syntax_names: &["Go"],
        has_symbols: true,
    },
    LanguageSpec {
        language: Language::Gdscript,
        id: "gdscript",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["gdscript", "gd"],
        extensions: &["gd"],
        file_names: &[],
        syntax_names: &["GDScript"],
        has_symbols: false,
    },
    LanguageSpec {
        language: Language::Java,
        id: "java",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["java"],
        extensions: &["java"],
        file_names: &[],
        syntax_names: &["Java"],
        has_symbols: true,
    },
    LanguageSpec {
        language: Language::JavaScript,
        id: "javascript",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["javascript", "js"],
        extensions: &["js", "mjs", "cjs"],
        file_names: &[],
        syntax_names: &["JavaScript"],
        has_symbols: true,
    },
    LanguageSpec {
        language: Language::Jsx,
        id: "jsx",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["jsx"],
        extensions: &["jsx"],
        file_names: &[],
        syntax_names: &[],
        has_symbols: true,
    },
    LanguageSpec {
        language: Language::Html,
        id: "html",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["html", "htm"],
        extensions: &["html", "htm"],
        file_names: &[],
        syntax_names: &["HTML"],
        has_symbols: false,
    },
    LanguageSpec {
        language: Language::Json,
        id: "json",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["json"],
        extensions: &["json"],
        file_names: &["package-lock.json", "composer.lock"],
        syntax_names: &["JSON"],
        has_symbols: false,
    },
    LanguageSpec {
        language: Language::Xml,
        id: "xml",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["xml"],
        extensions: &["xml", "xsd", "xsl", "xslt"],
        file_names: &[],
        syntax_names: &["XML"],
        has_symbols: false,
    },
    LanguageSpec {
        language: Language::Yaml,
        id: "yaml",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["yaml", "yml"],
        extensions: &["yaml", "yml"],
        file_names: &[],
        syntax_names: &["YAML"],
        has_symbols: false,
    },
    LanguageSpec {
        language: Language::Just,
        id: "just",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["just", "justfile"],
        extensions: &["just"],
        file_names: &["justfile", "Justfile", ".justfile"],
        syntax_names: &["Just"],
        has_symbols: true,
    },
    LanguageSpec {
        language: Language::Kconfig,
        id: "kconfig",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["kconfig"],
        extensions: &[],
        file_names: &["Kconfig"],
        syntax_names: &["Kconfig"],
        has_symbols: true,
    },
    LanguageSpec {
        language: Language::Make,
        id: "make",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["make", "makefile"],
        extensions: &["mk", "mak", "make"],
        file_names: &["Makefile", "makefile", "GNUmakefile"],
        syntax_names: &["Makefile"],
        has_symbols: true,
    },
    LanguageSpec {
        language: Language::Latex,
        id: "latex",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["latex", "tex"],
        extensions: &["tex", "ltx", "latex"],
        file_names: &[],
        syntax_names: &["LaTeX", "TeX"],
        has_symbols: false,
    },
    LanguageSpec {
        language: Language::Lua,
        id: "lua",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["lua"],
        extensions: &["lua"],
        file_names: &[],
        syntax_names: &["Lua"],
        has_symbols: false,
    },
    LanguageSpec {
        language: Language::Markdown,
        id: "markdown",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["markdown", "md"],
        extensions: &["md", "markdown", "mdown", "mkd"],
        file_names: &[],
        syntax_names: &["Markdown"],
        has_symbols: true,
    },
    LanguageSpec {
        language: Language::Meson,
        id: "meson",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["meson"],
        extensions: &[],
        file_names: &["meson.build", "meson_options.txt"],
        syntax_names: &["Meson"],
        has_symbols: false,
    },
    LanguageSpec {
        language: Language::Nix,
        id: "nix",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["nix"],
        extensions: &["nix"],
        file_names: &["flake.lock"],
        syntax_names: &["Nix"],
        has_symbols: false,
    },
    LanguageSpec {
        language: Language::Perl,
        id: "perl",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["perl", "pl", "pm"],
        extensions: &["pl", "pm", "t"],
        file_names: &[],
        syntax_names: &["Perl"],
        has_symbols: false,
    },
    LanguageSpec {
        language: Language::Python,
        id: "python",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["python", "py"],
        extensions: &["py", "pyw"],
        file_names: &[],
        syntax_names: &["Python"],
        has_symbols: true,
    },
    LanguageSpec {
        language: Language::Php,
        id: "php",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["php"],
        extensions: &["php", "php3", "php4", "php5", "phtml"],
        file_names: &[],
        syntax_names: &["PHP"],
        has_symbols: true,
    },
    LanguageSpec {
        language: Language::Puppet,
        id: "puppet",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["puppet", "pp"],
        extensions: &["pp"],
        file_names: &["Puppetfile"],
        syntax_names: &["Puppet"],
        has_symbols: false,
    },
    LanguageSpec {
        language: Language::Ruby,
        id: "ruby",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["ruby", "rb"],
        extensions: &["rb", "rake", "gemspec"],
        file_names: &["Gemfile", "Rakefile"],
        syntax_names: &["Ruby"],
        has_symbols: true,
    },
    LanguageSpec {
        language: Language::Riscv,
        id: "riscv",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["riscv", "risc-v", "riscv64"],
        extensions: &["riscv"],
        file_names: &[],
        syntax_names: &["RISC-V"],
        has_symbols: false,
    },
    LanguageSpec {
        language: Language::Rust,
        id: "rust",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["rust", "rs"],
        extensions: &["rs"],
        file_names: &[],
        syntax_names: &["Rust"],
        has_symbols: true,
    },
    LanguageSpec {
        language: Language::Swift,
        id: "swift",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["swift"],
        extensions: &["swift"],
        file_names: &[],
        syntax_names: &["Swift"],
        has_symbols: true,
    },
    LanguageSpec {
        language: Language::Sql,
        id: "sql",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["sql"],
        extensions: &["sql"],
        file_names: &[],
        syntax_names: &["SQL"],
        has_symbols: false,
    },
    LanguageSpec {
        language: Language::TypeScript,
        id: "typescript",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["typescript", "ts"],
        extensions: &["ts", "mts", "cts"],
        file_names: &[],
        syntax_names: &["TypeScript"],
        has_symbols: true,
    },
    LanguageSpec {
        language: Language::Tsx,
        id: "tsx",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["tsx"],
        extensions: &["tsx"],
        file_names: &[],
        syntax_names: &[],
        has_symbols: true,
    },
    LanguageSpec {
        language: Language::Toml,
        id: "toml",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["toml"],
        extensions: &["toml"],
        file_names: &["Cargo.lock"],
        syntax_names: &["TOML"],
        has_symbols: false,
    },
    LanguageSpec {
        language: Language::Typst,
        id: "typst",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["typst", "typ"],
        extensions: &["typ"],
        file_names: &[],
        syntax_names: &["Typst"],
        has_symbols: false,
    },
    LanguageSpec {
        language: Language::Vimscript,
        id: "vimscript",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["vimscript", "vim", "viml"],
        extensions: &["vim", "vimrc", "gvimrc"],
        file_names: &[".vimrc", ".gvimrc"],
        syntax_names: &["VimL", "Vimscript"],
        has_symbols: true,
    },
    LanguageSpec {
        language: Language::Zig,
        id: "zig",
        engine: Some(AnalysisEngine::TreeSitter),
        aliases: &["zig"],
        extensions: &["zig", "zon"],
        file_names: &["build.zig", "build.zig.zon"],
        syntax_names: &["Zig"],
        has_symbols: false,
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DocumentKind {
    Source,
    Text,
}

impl Serialize for Language {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.id())
    }
}

pub(crate) fn detect_language(path: &Path, text: &str) -> (Language, Option<String>) {
    if let Some(language) = detect_by_path(path) {
        return (language, None);
    }

    let syntax_set = {
        static SYNTAX_SET: OnceLock<SyntaxSet> = OnceLock::new();
        SYNTAX_SET.get_or_init(SyntaxSet::load_defaults_newlines)
    };
    let syntax = detect_syntax(syntax_set, path, text);

    let Some(syntax) = syntax else {
        return (Language::Unknown, None);
    };

    (
        LANGUAGE_SPECS
            .iter()
            .find_map(|spec| {
                spec.syntax_names
                    .contains(&syntax.name.as_str())
                    .then_some(spec.language)
            })
            .unwrap_or(Language::Unknown),
        Some(syntax.name.clone()),
    )
}

fn detect_syntax<'a>(
    syntax_set: &'a SyntaxSet,
    path: &Path,
    text: &str,
) -> Option<&'a SyntaxReference> {
    if let Some(name) = path.file_name().and_then(|name| name.to_str())
        && let Some(syntax) = syntax_set.find_syntax_by_extension(name)
    {
        return Some(syntax);
    }
    if let Some(ext) = path.extension().and_then(|ext| ext.to_str())
        && let Some(syntax) = syntax_set.find_syntax_by_extension(ext)
    {
        return Some(syntax);
    }
    let first_line = text.lines().next().unwrap_or("");
    syntax_set.find_syntax_by_first_line(first_line)
}

pub(crate) fn detect_by_path(path: &Path) -> Option<Language> {
    let file_name = path.file_name()?.to_str()?;
    LANGUAGE_SPECS
        .iter()
        .find_map(|spec| {
            spec.file_names
                .contains(&file_name)
                .then_some(spec.language)
        })
        .or_else(|| {
            let extension = path.extension()?.to_str()?;
            LANGUAGE_SPECS.iter().find_map(|spec| {
                spec.extensions
                    .iter()
                    .any(|&ext| ext.eq_ignore_ascii_case(extension))
                    .then_some(spec.language)
            })
        })
}

pub(crate) fn normalize_source_text(mut text: String) -> String {
    if text.starts_with('\u{feff}') {
        text.drain(..'\u{feff}'.len_utf8());
    }
    if !text.contains('\r') {
        return text;
    }
    text.replace("\r\n", "\n").replace('\r', "\n")
}
