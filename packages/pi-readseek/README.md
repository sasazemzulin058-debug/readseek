# pi-readseek

`pi-readseek` adds ReadSeek's anchored file tools, structural search, and symbol
navigation to Pi. Built-in tools remain unchanged unless `replacedTools` maps
`read`, `edit`, `write`, or `grep` to the ReadSeek implementation.

## Installation

```sh
pi install npm:pi-readseek
```

The extension depends on `@jarkkojs/readseek`, which installs the native binary on
supported platforms.

## Tools

- `readSeek_read`: reads anchored text by range, symbol, or structural map; it can
  also read or analyze images and PDFs.
- `readSeek_edit`: applies hash-verified edits to existing text files.
- `readSeek_write`: creates or replaces complete files and returns anchors.
- `readSeek_grep`: searches text and returns edit-ready anchors.
- `readSeek_search`: searches code with structural AST patterns.
- `readSeek_def`, `readSeek_refs`, `readSeek_hover`: navigate symbols.
- `readSeek_rename`: applies binding-aware renames by default. Set `apply: false`
  for a dry run; workspace matches are name-based where binding support is unavailable.
- `readSeek_check`: reports parser errors and missing syntax.
- `readSeek_view`: indexes a PDF or narrows an existing index by page, node, kind,
  or depth.

## Settings

Add an optional `readseek` section to `~/.pi/agent/settings.json` (global) or
`.pi/settings.json` (project). Project settings take precedence. Defaults:

```json
{
  "readseek": {
    "replacedTools": [],
    "imageMode": "auto",
    "syntaxValidation": "warn",
    "timeoutMs": 120000,
    "grep": {
      "maxLines": 2000,
      "maxBytes": 51200
    },
    "display": {
      "read": "compact",
      "grep": "compact",
      "edit": "expanded",
      "write": "expanded"
    }
  }
}
```

- `replacedTools`: built-in tools backed by ReadSeek. Valid values are `"read"`,
  `"edit"`, `"write"`, and `"grep"`.
- `imageMode`: `"auto"` exposes `none`, `all`, `ocr`, `caption`, and `objects`;
  `"on"` omits `none`; `"off"` skips images and PDFs. Omitting `image` also skips
  visual files.
- `syntaxValidation`: syntax-regression handling for `readSeek_edit`. `"warn"`
  writes with a warning, `"block"` aborts, and `"off"` disables the check.
- `timeoutMs`: ReadSeek command timeout in milliseconds.
- `grep.maxLines` and `grep.maxBytes`: output limits for `readSeek_grep`. Values
  above the defaults are clamped.
- `display`: default display state (`"compact"` or `"expanded"`) per tool for `read`, `grep`, `edit`, and `write`.

## Licensing

`pi-readseek` is licensed under `Apache-2.0`; see [LICENSE](LICENSE).
`@jarkkojs/readseek` is licensed separately under
`Apache-2.0 AND LGPL-2.1-or-later`.
