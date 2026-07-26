# readseek (Android / Termux ARM64 Edition)

[![Android ARM64 Build & Release](https://github.com/sasazemzulin058-debug/readseek/actions/workflows/build-release.yml/badge.svg)](https://github.com/sasazemzulin058-debug/readseek/actions/workflows/build-release.yml)

Custom fork of [`jarkkojs/readseek`](https://github.com/jarkkojs/readseek) tailored for **Android / Termux ARM64 (`aarch64-linux-android`)**.

Includes prebuilt Rust `readseek` binaries for Android NDK r27c and prebuilt `pi-readseek` extension tarballs for Pi harness agent.

---

## 📱 Android (Termux) Quickstart

Run the one-line installer in Termux:

```bash
curl -fsSL https://raw.githubusercontent.com/sasazemzulin058-debug/readseek/main/install-android.sh | sh
```

### What it installs:
1. **Prebuilt `readseek` CLI binary** into `${PREFIX}/bin/readseek` (Android `aarch64-linux-android`).
2. **`@sasazemzulin058-debug/readseek-android-arm64`** native npm package into `${HOME}/.pi/agent/npm`.
3. **`pi-readseek`** TS extension patched with native `"android-arm64"` platform mapping into `${HOME}/.pi/agent/npm`.

---

## 🔧 Termux / Android Fixes in this Fork

- **Native `aarch64-linux-android` Rust Compilation**: Built with Android NDK r27c using `aarch64-linux-android24-clang`.
- **First-class `android-arm64` Platform Mapping**: Added `@sasazemzulin058-debug/readseek-android-arm64` directly to `READSEEK_PLATFORM_PACKAGES` in `packages/pi-readseek/src/readseek-client.ts`.
- **Built-in `dist/index.js` Export**: Updated `package.json` build scripts to compile ESM `dist/index.js` directly via Bun.
- **Robust Termux Installer**: Portable `install-android.sh` using `${PREFIX}/tmp` and `${HOME}/.pi/agent/npm` with `trap` cleanup and `curl` retries.

---

## 🚀 Features & Usage

`readseek` provides anchored file reading, hashing (`LINE:HASH`), structural maps, and AST search for Pi extensions.

### Common CLI commands:
```bash
readseek detect src/main.rs
readseek read src/main.rs:10 --end 20
readseek map src/main.rs
readseek check src/main.rs
readseek identify src/main.rs:42 --column 8
readseek def src run --language rust
readseek refs src main --language rust
readseek search src 'fn $NAME() { $$$BODY }' --language rust
```

### Pi Extension Usage:
After installing via `install-android.sh`, `pi-readseek` tools are automatically registered:
- `readSeek_read`: Read anchored text and file hashes.
- `readSeek_edit`: Hash-safe line editing.
- `readSeek_grep`: Fast regex search returning `LINE:HASH` anchors.
- `readSeek_search`: AST pattern searching.

---

## 📜 License

- `readseek` CLI: LGPL 2.1+
- `pi-readseek`: Apache 2.0
