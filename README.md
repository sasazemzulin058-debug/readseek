# readseek

<img alt="Screencast of readseek captioning an image" src="screencast.gif" width="1024">


`readseek` reads source files and PDFs for scripts, editors, and coding agents.
It returns compact JSON with stable line/hash anchors, symbol maps, parse
diagnostics, AST matches, references, and rename plans.

The screencast above was recorded with a very modest AMD Ryzen 5 PRO 4650 laptop
CPU, and demonstrates the tailored inference engine for Qwen3-VL-2B.

## Installation

Install the npm wrapper with a prebuilt binary:

```sh
npm install -g @jarkkojs/readseek
```

Or install the native program from crates.io:

```sh
cargo install readseek
```

Prebuilt binaries are available for macOS ARM64; Linux ARM64 and x64; and Windows
x64. The Linux binaries are statically linked with musl.

To build and install from source:

```sh
make install
```

## Plugins

### Pi extension

The bundled [pi-readseek extension](packages/pi-readseek/README.md) adds ReadSeek's
anchored file and structural-code tools to Pi:

```sh
pi install npm:pi-readseek
```

### OpenCode plugin

The [opencode-readseek plugin](packages/opencode-readseek/README.md) adds the same
tools to OpenCode. Add it to `opencode.json`:

```json
{
  "plugin": ["opencode-readseek"]
}
```

### Vim plugin

The bundled `readseek.vim` plugin provides anchored reads, structural navigation,
parse diagnostics, search, references, and renames. It requires Vim9 with `+job`,
`+channel`, `+timers`, `+popupwin`, and `+textprop`; Neovim is not supported.

Install it with a Vim plugin manager, such as `Plug 'jarkkojs/readseek'`, then run
`:ReadSeekInstall` to install the matching prebuilt binary. Downloads are explicit
by default.

## Common commands

```sh
readseek detect src/main.rs
readseek read src/main.rs:10 --end 20
readseek read report.pdf --page 3
readseek view report.pdf --page 3
readseek map src/main.rs
readseek check src/main.rs
readseek symbol src/main.rs:run --name
readseek identify src/main.rs:42 --column 8
readseek def src run --language rust --format plain
readseek refs src main --language rust --format plain
readseek search src 'fn $NAME() { $$$BODY }' --language rust
readseek rename src/main.rs --line 42 --column 8 --to renamed
```

Global options must precede the command. For example, write JSON to a file with:

```sh
readseek --output result.json detect src/main.rs
```

Prefix a target with `stdin:` to analyze an unsaved buffer while retaining a path
for language detection and cursor addressing. This works with `detect`, `read`,
`map`, `check`, `symbol`, and `identify`:

```sh
printf '%s\n' 'fn main() {}' | readseek identify stdin:scratch.rs:1 --column 4
```

## Images and PDFs

`detect` reports image metadata and PDF page counts. `read` returns bounded base64
images by default; select local analysis with `--vision-mode`:

```sh
readseek read photo.jpg                         # default: bounded base64 image
readseek read photo.jpg --vision-mode caption   # detailed natural-language caption
readseek read photo.jpg --vision-mode objects   # object labels + bounding boxes
readseek read photo.jpg --vision-mode ocr       # extracted text
readseek read photo.jpg --vision-mode all       # caption, objects, and OCR in one pass
```

Vision analysis uses the `low` level by default. Start at `low`, then increase to
`medium` or `high` only when additional detail is needed:

```sh
readseek read photo.jpg --vision-mode caption --vision-level low
readseek read scan.png --vision-mode ocr --vision-level high
```

Set `RUST_LOG=tracing` to emit vision cache and inference traces on standard error.

Run `cargo bench --features vision-bench --bench vision` for data-driven comparisons
defined in `benches/vision.txt`.

Set `READSEEK_VISION_THREADS` to a positive integer to override Rayon's worker
count. More workers may improve throughput at the cost of CPU and memory.

PDF reads return page-tagged Markdown and embedded images. Use `--page` to select a
page; `--vision-mode` applies to each embedded image. After `readseek init`, `view` creates
or reuses a structural PDF index that can be narrowed by page, node, kind, or depth.
Line/hash suffixes, `--end`, `--limit`, and `--language` do not apply to visual files.

The Qwen3-VL-2B Q4_K_M model and Q8_0 multimodal projector download lazily to the user
cache and are checksum-verified. Inference uses ReadSeek's built-in CPU engine and
can be slow.

## Cache

`readseek init [path]` creates `.readseek/maps/` and `.readseek/def-index/`.
Map-dependent commands update them on demand and find `.readseek/` by walking up
from the target; `--readseek-dir` selects one explicitly. PDF indexes and extracted
assets live in `.readseek/documents/`, while image analysis results are cached under
`.readseek/vision/`. Image cache entries are level-specific.

## Documentation

For the complete CLI reference:

```sh
man ./man/man1/readseek.1
```

Pass `--help` to any command for command-specific usage.

## Licensing

1. `opencode-readseek`: Apache 2.0
2. `pi-readseek`: Apache 2.0
3. `readseek`: LGPL 2.1+
4. `readseek.vim`: Apache 2.0

### Third Party Attribution

1. Qwen3-VL-2B: Apache 2.0
   - Downloaded on first use at run-time. The model is not distributed together
     with ReadSeek.
2. [Dwarf Seek 4](https://github.com/antirez/ds4): MIT
   - Q4\_K and Q6\_K block decoding and dot-product implementation are derived
     works of DS4.
