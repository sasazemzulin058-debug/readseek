// SPDX-License-Identifier: Apache-2.0
// Copyright (C) Jarkko Sakkinen 2026

import { lstat, mkdir, open, readFile, readlink, realpath, stat } from "node:fs/promises";
import { createRequire } from "node:module";
import { homedir } from "node:os";
import path from "node:path";

import { tool, type Plugin, type PluginOptions, type ToolAttachment, type ToolContext } from "@opencode-ai/plugin";

const require = createRequire(import.meta.url);
const MAX_OUTPUT_BYTES = 32 * 1024 * 1024;
const DEFAULT_READ_LIMIT = 2000;
const EDIT_RESULT_READ_LIMIT = 40;
const MAX_DOCUMENT_OUTPUT_BYTES = 256 * 1024;
const MAX_DOCUMENT_OUTPUT_LINES = 2000;
const TOOL_POLICY = [
  "ReadSeek tool policy:",
  "- Prefer readseek_* tools over OpenCode built-ins whenever they can do the job.",
  "- Prefer readseek_read for text file reads; its LINE:HASH anchors are required by readseek_edit.",
  "- Prefer readseek_grep for text/regex search and readseek_search for syntax-aware code shapes.",
  "- Prefer readseek_map, readseek_def, readseek_refs, and readseek_hover for symbol navigation.",
  "- Prefer readseek_edit for existing text files, readseek_write for whole-file creation or replacement, and readseek_rename for symbol renames.",
  "- Do not use built-in read, edit, write, grep, or apply_patch when the corresponding ReadSeek tool can perform the change.",
  "- Use readseek_check after source edits for a quick syntax check.",
].join("\n");
const PREFERRED_TOOL_DESCRIPTIONS: Record<string, string> = {
  readseek_read: "Preferred file reader; returns LINE:HASH anchors required by readseek_edit.",
  readseek_edit: "Preferred tool for editing existing text files with verified LINE:HASH anchors.",
  readseek_write: "Preferred tool for creating or replacing a complete text file.",
  readseek_rename: "Preferred tool for renaming code symbols safely across their resolved bindings.",
  readseek_grep: "Preferred text/regex search; returns edit-ready LINE:HASH anchors.",
  readseek_search: "Preferred syntax-aware code search for AST shapes and call sites.",
  readseek_map: "Preferred structural outline of symbols in a source file.",
  readseek_def: "Preferred symbol definition lookup.",
  readseek_refs: "Preferred identifier usage finder before rename or delete.",
  readseek_hover: "Preferred cursor token and enclosing-symbol inspector.",
  readseek_view: "Preferred PDF structure and content viewer.",
  readseek_check: "Preferred quick syntax check after source edits.",
};
const FALLBACK_TOOL_DESCRIPTIONS: Record<string, string> = {
  read: "Fallback only. Prefer readseek_read for text files so edits can use LINE:HASH anchors.",
  edit: "Fallback only. Prefer readseek_edit with anchors from readseek_read/readseek_grep/readseek_search.",
  write: "Fallback only. Prefer readseek_write so the result returns LINE:HASH anchors.",
  grep: "Fallback only. Prefer readseek_grep for project text search with edit-ready anchors.",
  apply_patch: "Fallback only. Prefer readseek_edit for existing text files.",
};
const READSEEK_PLATFORM_PACKAGES: Record<string, string> = {
  "darwin-arm64": "@jarkkojs/readseek-darwin-arm64",
  "linux-arm64": "@jarkkojs/readseek-linux-arm64",
  "linux-x64": "@jarkkojs/readseek-linux-x64",
  "win32-x64": "@jarkkojs/readseek-win32-x64",
};

type RenamePlan = {
  summary: string;
  files: Set<string>;
};

type PresentationKind = "read" | "view" | "map" | "search" | "grep" | "def" | "refs" | "hover" | "rename" | "edit" | "write" | "check";

type Presentation = {
  title: string;
  metadata: Record<string, number>;
};

type RenderedToolResult = {
  output: string;
  attachments?: ToolAttachment[];
};

type ImagePolicy = "on" | "auto" | "off";
type ImageMode = "none" | "all" | "ocr" | "caption" | "objects";
type DocumentNodeKind = "artifact" | "footer" | "header" | "heading" | "marginal_label" | "page" | "page_number" | "paragraph" | "section" | "structural_section";

const DOCUMENT_NODE_KINDS: [DocumentNodeKind, ...DocumentNodeKind[]] = [
  "artifact",
  "footer",
  "header",
  "heading",
  "marginal_label",
  "page",
  "page_number",
  "paragraph",
  "section",
  "structural_section",
];

function resolveImagePolicy(options: PluginOptions | undefined): ImagePolicy {
  const value = options?.imageMode;
  if (value === undefined) return "auto";
  if (value === "on" || value === "auto" || value === "off") return value;
  throw new Error('opencode-readseek imageMode must be "on", "auto", or "off"');
}

function isVisualFile(value: unknown): boolean {
  const output = record(value);
  const type = output.type;
  return typeof output.width === "number" || type === "application/pdf";
}

function isPdfFile(value: unknown): boolean {
  return record(value).format === "pdf";
}

class SessionAnchors {
  #pathsBySession = new Map<string, Set<string>>();
  #renamePlans = new Map<string, RenamePlan>();

  mark(sessionID: string, filePath: string): void {
    let paths = this.#pathsBySession.get(sessionID);
    if (!paths) {
      paths = new Set<string>();
      this.#pathsBySession.set(sessionID, paths);
    }
    paths.add(filePath);
  }

  forget(filePath: string): void {
    const absolutePath = path.resolve(filePath);
    for (const paths of this.#pathsBySession.values()) paths.delete(absolutePath);
    for (const [sessionID, plan] of this.#renamePlans) {
      if (plan.files.has(absolutePath)) this.#renamePlans.delete(sessionID);
    }
  }

  deleteSession(sessionID: string): void {
    this.#pathsBySession.delete(sessionID);
    this.#renamePlans.delete(sessionID);
  }

  planRename(sessionID: string, output: unknown): void {
    if (!output || typeof output !== "object") return;
    const record = output as Record<string, unknown>;
    if (record.applied === true) {
      this.#renamePlans.delete(sessionID);
      return;
    }
    const { file, old_name: oldName, new_name: newName, others } = record;
    if (typeof file !== "string" || typeof oldName !== "string" || typeof newName !== "string") return;

    const files = new Set([path.resolve(file)]);
    if (Array.isArray(others)) {
      for (const item of others) {
        if (!item || typeof item !== "object") continue;
        const otherFile = (item as Record<string, unknown>).file;
        if (typeof otherFile === "string") files.add(path.resolve(otherFile));
      }
    }
    this.#renamePlans.set(sessionID, { summary: `${oldName} -> ${newName}`, files });
  }

  render(sessionID: string): string | undefined {
    const paths = this.#pathsBySession.get(sessionID);
    const sections: string[] = [];
    if (paths?.size) {
      sections.push(`## ReadSeek Anchors\nThe following files have fresh LINE:HASH anchors:\n${[...paths]
      .sort()
      .map((filePath) => `- ${filePath}`)
      .join("\n")}`);
    }
    const renamePlan = this.#renamePlans.get(sessionID);
    if (renamePlan) sections.push(`## Pending ReadSeek Rename Plan\n- ${renamePlan.summary}`);
    return sections.length === 0 ? undefined : sections.join("\n\n");
  }
}

function resolvePath(directory: string, filePath: string): string {
  return path.resolve(directory, filePath);
}

function containsPath(directory: string, filePath: string): boolean {
  const relative = path.relative(directory, filePath);
  return relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative));
}

async function resolveSymlinks(filePath: string): Promise<string> {
  let unresolved = path.resolve(filePath);
  let symlinks = 0;

  while (true) {
    const parsed = path.parse(unresolved);
    const parts = unresolved.slice(parsed.root.length).split(path.sep).filter(Boolean);
    let resolved = parsed.root;
    let restart = false;

    for (let index = 0; index < parts.length; index++) {
      const candidate = path.join(resolved, parts[index]);
      const info = await lstat(candidate);
      if (!info.isSymbolicLink()) {
        resolved = candidate;
        continue;
      }

      symlinks++;
      if (symlinks > 40) {
        const error = new Error(`too many symbolic links: ${filePath}`) as NodeJS.ErrnoException;
        error.code = "ELOOP";
        throw error;
      }

      const target = await readlink(candidate);
      unresolved = path.resolve(path.dirname(candidate), target, ...parts.slice(index + 1));
      restart = true;
      break;
    }

    if (!restart) return resolved;
  }
}

async function portableRealpath(filePath: string): Promise<string> {
  try {
    return await realpath(filePath);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EACCES") throw error;
    return resolveSymlinks(filePath);
  }
}

async function canonicalPath(filePath: string): Promise<string> {
  const missing: string[] = [];
  let existing = filePath;
  while (true) {
    try {
      return path.join(await portableRealpath(existing), ...missing.reverse());
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      const parent = path.dirname(existing);
      if (parent === existing) throw error;
      missing.push(path.basename(existing));
      existing = parent;
    }
  }
}

async function authorizeExternal(context: ToolContext, filePath: string): Promise<void> {
  const [accessPath, directory, worktree] = await Promise.all([
    canonicalPath(filePath),
    canonicalPath(context.directory),
    context.worktree === "/" ? Promise.resolve("/") : canonicalPath(context.worktree),
  ]);
  if (containsPath(directory, accessPath) || (worktree !== "/" && containsPath(worktree, accessPath))) {
    return;
  }

  const info = await stat(accessPath).catch(() => undefined);
  const parentDir = info?.isDirectory() ? accessPath : path.dirname(accessPath);
  const pattern = path.join(parentDir, "*").replaceAll("\\", "/");
  await context.ask({
    permission: "external_directory",
    patterns: [pattern],
    always: [pattern],
    metadata: { filepath: filePath, parentDir },
  });
}

async function rejectSymlinkMutation(filePath: string): Promise<void> {
  const info = await lstat(filePath).catch((error: NodeJS.ErrnoException) => {
    if (error.code === "ENOENT") return undefined;
    throw error;
  });
  if (info?.isSymbolicLink()) throw new Error(`refusing to modify symbolic link: ${filePath}`);
}

async function authorizeRead(context: ToolContext, filePath: string): Promise<void> {
  await authorizeExternal(context, filePath);
  await context.ask({
    permission: "read",
    patterns: [path.relative(context.worktree, filePath).replaceAll("\\", "/")],
    always: ["*"],
    metadata: {},
  });
}

async function authorizeSearch(context: ToolContext, filePath: string, pattern: string): Promise<void> {
  await context.ask({
    permission: "grep",
    patterns: [pattern],
    always: ["*"],
    metadata: { pattern, path: filePath },
  });
  await authorizeExternal(context, filePath);
}

async function authorizeEdit(
  context: ToolContext,
  filePaths: string[],
  diff = "",
  authorizeExternalPaths = true,
): Promise<void> {
  if (authorizeExternalPaths) {
    for (const filePath of filePaths) await authorizeExternal(context, filePath);
  }
  const patterns = filePaths.map((filePath) => path.relative(context.worktree, filePath).replaceAll("\\", "/"));
  await context.ask({
    permission: "edit",
    patterns,
    always: ["*"],
    metadata: { filepath: filePaths.join(", "), diff },
  });
}

function optionalFlag(args: string[], enabled: boolean | undefined, flag: string): void {
  if (enabled) args.push(flag);
}

function readSeekBinaryPath(): string {
  const platform = `${process.platform}-${process.arch}`;
  const platformPackage = READSEEK_PLATFORM_PACKAGES[platform];
  if (!platformPackage) {
    throw new Error(`@jarkkojs/readseek ships no binary for ${platform}`);
  }

  const readseekPackageDir = path.dirname(require.resolve("@jarkkojs/readseek/package.json"));
  const packageJson = require.resolve(`${platformPackage}/package.json`, { paths: [readseekPackageDir] });
  return path.join(path.dirname(packageJson), "bin", process.platform === "win32" ? "readseek.exe" : "readseek");
}

async function runReadSeekRaw(
  context: ToolContext,
  args: string[],
  options: { cancelable?: boolean } = {},
): Promise<string> {
  context.abort.throwIfAborted();
  const cancelable = options.cancelable !== false;
  const spawnOptions = {
    cwd: context.directory,
    maxBuffer: MAX_OUTPUT_BYTES,
    stderr: "pipe" as const,
    stdout: "pipe" as const,
  };
  const child = Bun.spawn(
    [readSeekBinaryPath(), ...args],
    cancelable
      ? { ...spawnOptions, killSignal: "SIGKILL" as const, signal: context.abort }
      : spawnOptions,
  );
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
    child.exited,
  ]);
  if (cancelable) context.abort.throwIfAborted();
  if (exitCode !== 0) throw new Error(stderr.trim() || `readseek exited with status ${exitCode}`);
  return stdout;
}

async function runReadSeek(
  context: ToolContext,
  args: string[],
  options: { cancelable?: boolean } = {},
): Promise<unknown> {
  const stdout = await runReadSeekRaw(context, args, options);
  try {
    return JSON.parse(stdout) as unknown;
  } catch {
    throw new Error(`readseek returned invalid JSON: ${stdout.trim()}`);
  }
}

let readSeekCacheInit: Promise<string> | undefined;

function readSeekCacheDir(): string {
  const xdgCacheHome = process.env.XDG_CACHE_HOME?.trim();
  const cacheHome = xdgCacheHome || path.join(homedir(), ".cache");
  return path.join(cacheHome, "opencode", "readseek");
}

async function ensureReadSeekCache(context: ToolContext): Promise<string> {
  const cacheDir = readSeekCacheDir();
  try {
    const info = await stat(cacheDir);
    if (!info.isDirectory()) throw new Error(`readseek cache is not a directory: ${cacheDir}`);
    return cacheDir;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }

  readSeekCacheInit ??= runReadSeekRaw(context, ["--readseek-dir", cacheDir, "init"])
    .then(() => cacheDir)
    .finally(() => {
      readSeekCacheInit = undefined;
    });
  return readSeekCacheInit;
}

function render(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function collectAnchoredFiles(value: unknown, files: Set<string>): void {
  if (Array.isArray(value)) {
    for (const item of value) collectAnchoredFiles(item, files);
    return;
  }
  if (!value || typeof value !== "object") return;

  const output = value as Record<string, unknown>;
  const hashlines = output.hashlines;
  if (
    typeof output.file === "string" &&
    Array.isArray(hashlines) &&
    hashlines.some((line) => typeof record(line).line === "number" && typeof record(line).hash === "string")
  ) files.add(output.file);
  for (const item of Object.values(output)) collectAnchoredFiles(item, files);
}

type AnchorEdit =
  | { set_line: { anchor: string; new_text: string } }
  | { replace_lines: { start_anchor: string; end_anchor: string; new_text: string } }
  | { insert_after: { anchor: string; new_text: string } };

type TextSnapshot = {
  exists: boolean;
  content: string;
};

const fileMutationTails = new Map<string, Promise<void>>();
const MAX_PERMISSION_DIFF_BYTES = 32 * 1024;

async function withFileMutationQueue<T>(filePath: string, operation: () => Promise<T>): Promise<T> {
  const key = path.resolve(filePath);
  const previous = fileMutationTails.get(key);
  let release = () => {};
  const current = new Promise<void>((resolve) => {
    release = resolve;
  });
  fileMutationTails.set(key, current);
  if (previous) await previous;
  try {
    return await operation();
  } finally {
    release();
    if (fileMutationTails.get(key) === current) fileMutationTails.delete(key);
  }
}

function validateAnchorEdits(value: unknown): asserts value is AnchorEdit[] {
  if (!Array.isArray(value) || value.length === 0) throw new Error("edits must contain at least one edit");
  const fields = {
    set_line: ["anchor", "new_text"],
    replace_lines: ["end_anchor", "new_text", "start_anchor"],
    insert_after: ["anchor", "new_text"],
  } as const;
  for (const [index, item] of value.entries()) {
    if (!item || typeof item !== "object") throw new Error(`edits[${index}] must be an edit object`);
    const keys = Object.keys(item);
    const variant = keys[0] as keyof typeof fields;
    if (keys.length !== 1 || !Object.hasOwn(fields, variant)) {
      throw new Error(`edits[${index}] must contain exactly one of: set_line, replace_lines, insert_after`);
    }
    const payload = (item as Record<string, unknown>)[variant];
    if (!payload || typeof payload !== "object") throw new Error(`edits[${index}].${variant} must be an object`);
    const expectedFields = fields[variant];
    const actualFields = Object.keys(payload).sort();
    if (actualFields.length !== expectedFields.length || actualFields.some((field, fieldIndex) => field !== expectedFields[fieldIndex])) {
      throw new Error(`edits[${index}].${variant} contains invalid fields`);
    }
    const values = payload as Record<string, unknown>;
    for (const field of expectedFields) {
      if (typeof values[field] !== "string") {
        throw new Error(`edits[${index}].${variant}.${field} must be a string`);
      }
    }
    for (const field of expectedFields.filter((field) => field.endsWith("anchor"))) {
      parseAnchor(values[field] as string);
    }
  }
}

function parseAnchor(anchor: string): { line: number; hash: string } {
  const match = /^(\d+):([0-9a-fA-F]{3})$/.exec(anchor.trim());
  if (!match) throw new Error(`invalid LINE:HASH anchor: ${anchor}`);
  const line = Number(match[1]);
  if (line === 0) throw new Error(`anchor line must be greater than zero: ${anchor}`);
  return { line, hash: match[2].toLowerCase() };
}

function anchorRefs(edit: AnchorEdit): { line: number; hash: string }[] {
  if ("set_line" in edit) return [parseAnchor(edit.set_line.anchor)];
  if ("insert_after" in edit) return [parseAnchor(edit.insert_after.anchor)];
  return [parseAnchor(edit.replace_lines.start_anchor), parseAnchor(edit.replace_lines.end_anchor)];
}

async function verifyAnchors(context: ToolContext, filePath: string, edits: AnchorEdit[]): Promise<void> {
  const refs = new Map<number, string>();
  for (const edit of edits) {
    for (const ref of anchorRefs(edit)) {
      const previous = refs.get(ref.line);
      if (previous !== undefined && previous !== ref.hash) throw new Error(`conflicting hashes for line ${ref.line}`);
      refs.set(ref.line, ref.hash);
    }
  }
  for (const [line, expected] of refs) {
    const output = record(await runReadSeek(context, ["read", `${filePath}:${line}`, "--end", String(line)]));
    const hashline = items(output.hashlines).map(record)[0];
    const actual = hashline?.hash;
    if (typeof actual !== "string" || actual !== expected) {
      throw new Error(`stale anchor ${line}:${expected}; current hash is ${typeof actual === "string" ? actual : "unavailable"}`);
    }
  }
}

type EditableLine = {
  text: string;
  ending: "\r\n" | "\n" | "\r" | "";
};

function parseEditableLines(content: string): EditableLine[] {
  const lines: EditableLine[] = [];
  let start = 0;
  for (let index = 0; index < content.length; index++) {
    const character = content[index];
    if (character !== "\r" && character !== "\n") continue;
    const ending = character === "\r" && content[index + 1] === "\n" ? "\r\n" : character;
    lines.push({ text: content.slice(start, index), ending });
    if (ending === "\r\n") index++;
    start = index + 1;
  }
  if (start < content.length || content.length === 0) lines.push({ text: content.slice(start), ending: "" });
  return lines;
}

function detectLineEnding(lines: EditableLine[]): "\r\n" | "\n" | "\r" {
  const counts = new Map<"\r\n" | "\n" | "\r", number>([
    ["\r\n", 0],
    ["\n", 0],
    ["\r", 0],
  ]);
  for (const line of lines) {
    if (line.ending !== "") counts.set(line.ending, (counts.get(line.ending) ?? 0) + 1);
  }
  let selected: "\r\n" | "\n" | "\r" = "\n";
  for (const ending of ["\r\n", "\r"] as const) {
    if ((counts.get(ending) ?? 0) > (counts.get(selected) ?? 0)) selected = ending;
  }
  return selected;
}

function normalizeToLf(content: string): string {
  return content.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
}

function replacementLines(text: string): string[] {
  if (text === "") return [];
  return normalizeToLf(text).replace(/\n$/, "").split("\n");
}

function applyAnchorEdits(content: string, edits: AnchorEdit[]): string {
  const bom = content.startsWith("\uFEFF") ? "\uFEFF" : "";
  const lines = parseEditableLines(bom ? content.slice(1) : content);
  const newline = detectLineEnding(lines);
  const planned = edits.map((edit) => {
    if ("set_line" in edit) {
      const { line } = parseAnchor(edit.set_line.anchor);
      return { start: line - 1, deleteCount: 1, text: edit.set_line.new_text };
    }
    if ("insert_after" in edit) {
      const { line } = parseAnchor(edit.insert_after.anchor);
      return { start: line, deleteCount: 0, text: edit.insert_after.new_text };
    }
    const start = parseAnchor(edit.replace_lines.start_anchor).line;
    const end = parseAnchor(edit.replace_lines.end_anchor).line;
    if (end < start) throw new Error("replace_lines end anchor precedes start anchor");
    return { start: start - 1, deleteCount: end - start + 1, text: edit.replace_lines.new_text };
  });
  const ascending = [...planned].sort((left, right) => left.start - right.start || left.deleteCount - right.deleteCount);
  for (let index = 1; index < ascending.length; index++) {
    const previous = ascending[index - 1];
    const current = ascending[index];
    const previousEnd = previous.start + Math.max(previous.deleteCount, 1);
    if (current.start < previousEnd) throw new Error("anchored edits overlap or target the same location");
  }
  planned.sort((left, right) => right.start - left.start);
  for (const edit of planned) {
    if (edit.start < 0 || edit.start > lines.length || edit.start + edit.deleteCount > lines.length) {
      throw new Error("anchor line is outside the file");
    }
    const removed = lines.slice(edit.start, edit.start + edit.deleteCount);
    const replacement = replacementLines(edit.text).map((text): EditableLine => ({ text, ending: newline }));
    if (edit.deleteCount === 0 && edit.start === lines.length && replacement.length > 0) {
      const previous = lines.at(-1);
      const terminalEnding = previous?.ending ?? "";
      if (previous?.ending === "") previous.ending = newline;
      replacement[replacement.length - 1].ending = terminalEnding;
    } else if (edit.deleteCount > 0) {
      const terminalEnding = removed.at(-1)?.ending ?? "";
      if (replacement.length > 0) {
        replacement[replacement.length - 1].ending = terminalEnding;
      } else if (edit.start + edit.deleteCount === lines.length && edit.start > 0) {
        lines[edit.start - 1].ending = terminalEnding;
      }
    }
    lines.splice(edit.start, edit.deleteCount, ...replacement);
  }
  return bom + lines.map((line) => line.text + line.ending).join("");
}

async function readTextSnapshot(filePath: string): Promise<TextSnapshot> {
  try {
    return { exists: true, content: await readFile(filePath, "utf8") };
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return { exists: false, content: "" };
    throw error;
  }
}

async function writeHandle(handle: Awaited<ReturnType<typeof open>>, content: string): Promise<void> {
  const data = Buffer.from(content, "utf8");
  await handle.truncate(0);
  let offset = 0;
  while (offset < data.length) {
    const { bytesWritten } = await handle.write(data, offset, data.length - offset, offset);
    if (bytesWritten === 0) throw new Error("write made no progress");
    offset += bytesWritten;
  }
}

async function writeText(filePath: string, expected: TextSnapshot, content: string): Promise<void> {
  await mkdir(path.dirname(filePath), { recursive: true });
  if (!expected.exists) {
    let handle: Awaited<ReturnType<typeof open>>;
    try {
      handle = await open(filePath, "wx+");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "EEXIST") {
        throw new Error(`refusing to overwrite changed file: ${filePath}`);
      }
      throw error;
    }
    try {
      await writeHandle(handle, content);
    } finally {
      await handle.close();
    }
    return;
  }

  const pathInfo = await lstat(filePath);
  if (pathInfo.isSymbolicLink()) throw new Error(`refusing to modify symbolic link: ${filePath}`);
  const handle = await open(filePath, "r+");
  try {
    const handleInfo = await handle.stat();
    const currentPathInfo = await lstat(filePath);
    if (
      currentPathInfo.isSymbolicLink() ||
      currentPathInfo.dev !== handleInfo.dev ||
      currentPathInfo.ino !== handleInfo.ino
    ) {
      throw new Error(`refusing to overwrite changed file: ${filePath}`);
    }
    const current = (await handle.readFile()).toString("utf8");
    if (current !== expected.content) throw new Error(`refusing to overwrite changed file: ${filePath}`);
    if (current !== content) await writeHandle(handle, content);
  } finally {
    await handle.close();
  }
}

function renameFiles(value: unknown): string[] {
  const output = record(value);
  const files: string[] = [];
  if (typeof output.file === "string" && items(output.edits).length > 0) files.push(path.resolve(output.file));
  for (const other of items(output.others)) {
    const item = record(other);
    if (typeof item.file === "string" && items(item.edits).length > 0) files.push(path.resolve(item.file));
  }
  return files;
}

function diffLines(content: string): string[] {
  if (content === "") return [];
  const lines = normalizeToLf(content).split("\n");
  if (lines.at(-1) === "") lines.pop();
  return lines;
}

function diffRange(start: number, count: number): string {
  return count === 0 ? `${start},0` : `${start + 1},${count}`;
}

function utf8Prefix(value: string, maxBytes: number): string {
  const bytes = Buffer.from(value, "utf8");
  if (bytes.length <= maxBytes) return value;
  let end = maxBytes;
  const decoder = new TextDecoder("utf-8", { fatal: true });
  while (end > 0) {
    try {
      return decoder.decode(bytes.subarray(0, end));
    } catch {
      end--;
    }
  }
  return "";
}

function boundDocumentOutput(value: string): string {
  const lines = value.split("\n");
  let output = lines.length > MAX_DOCUMENT_OUTPUT_LINES
    ? lines.slice(0, MAX_DOCUMENT_OUTPUT_LINES).join("\n")
    : value;
  const truncated = output !== value || Buffer.byteLength(output, "utf8") > MAX_DOCUMENT_OUTPUT_BYTES;
  output = utf8Prefix(output, MAX_DOCUMENT_OUTPUT_BYTES);
  return truncated
    ? `${output}\n[… document view truncated; narrow it with page, node, kind, or depth]`
    : output;
}

function attachmentExtension(mime: string): string {
  switch (mime) {
    case "image/gif":
      return "gif";
    case "image/jpeg":
      return "jpg";
    case "image/png":
      return "png";
    case "image/webp":
      return "webp";
    default:
      return "bin";
  }
}

function imageAttachment(image: Record<string, unknown>, filename: string): ToolAttachment | undefined {
  if (image.encoding !== "base64" || typeof image.data !== "string" || typeof image.mime !== "string") {
    return undefined;
  }
  return {
    type: "file",
    mime: image.mime,
    url: `data:${image.mime};base64,${image.data}`,
    filename: `${filename}.${attachmentExtension(image.mime)}`,
  };
}

function formatReadResult(value: unknown): RenderedToolResult {
  const output = record(value);
  if (output.format === "pdf") {
    const attachments: ToolAttachment[] = [];
    const images = items(output.images).map((value, index) => {
      const image = record(value);
      const page = typeof image.page === "number" ? image.page : "unknown";
      const attachment = imageAttachment(image, `pdf-page-${page}-image-${index + 1}`);
      if (attachment) attachments.push(attachment);
      const sanitized = { ...image };
      delete sanitized.data;
      delete sanitized.encoding;
      return sanitized;
    });
    return {
      output: boundDocumentOutput(render({ ...output, images })),
      ...(attachments.length > 0 ? { attachments } : {}),
    };
  }

  const attachment = imageAttachment(output, "image");
  if (!attachment) return { output: render(value) };
  const sanitized = { ...output };
  delete sanitized.data;
  delete sanitized.encoding;
  return { output: render(sanitized), attachments: [attachment] };
}

function previewDiffLine(prefix: string, line: string): string {
  const value = `${prefix}${line}`;
  const maxBytes = 256;
  if (Buffer.byteLength(value) <= maxBytes) return value;
  return `${utf8Prefix(value, maxBytes - Buffer.byteLength("…"))}…`;
}

function sampleDiffLines(lines: string[], start: number, end: number, prefix: string): string[] {
  const sampleSize = 20;
  const count = end - start;
  if (count <= sampleSize * 2) {
    return lines.slice(start, end).map((line) => previewDiffLine(prefix, line));
  }
  return [
    ...lines.slice(start, start + sampleSize).map((line) => previewDiffLine(prefix, line)),
    ` ${count - sampleSize * 2} ${prefix === "-" ? "removed" : "added"} lines omitted`,
    ...lines.slice(end - sampleSize, end).map((line) => previewDiffLine(prefix, line)),
  ];
}

function simpleDiff(filePath: string, before: string, after: string): string {
  if (before === after) return "";
  const oldLines = diffLines(before);
  const newLines = diffLines(after);
  let prefix = 0;
  while (prefix < oldLines.length && prefix < newLines.length && oldLines[prefix] === newLines[prefix]) prefix++;
  let suffix = 0;
  while (
    suffix < oldLines.length - prefix &&
    suffix < newLines.length - prefix &&
    oldLines[oldLines.length - 1 - suffix] === newLines[newLines.length - 1 - suffix]
  ) suffix++;
  if (prefix === oldLines.length && prefix === newLines.length) {
    prefix = Math.max(0, prefix - 1);
    suffix = 0;
  }

  const oldChangeEnd = oldLines.length - suffix;
  const newChangeEnd = newLines.length - suffix;
  const oldStart = Math.max(0, prefix - 3);
  const newStart = Math.max(0, prefix - 3);
  const oldEnd = Math.min(oldLines.length, oldChangeEnd + 3);
  const newEnd = Math.min(newLines.length, newChangeEnd + 3);
  const safePath = utf8Prefix(filePath.replace(/[\r\n]/g, ""), 1024);
  const header = [
    `--- ${safePath}`,
    `+++ ${safePath}`,
    `@@ -${diffRange(oldStart, oldEnd - oldStart)} +${diffRange(newStart, newEnd - newStart)} @@`,
  ];
  const fullDiff = [...header];
  let fullDiffBytes = Buffer.byteLength(header.join("\n"));
  let complete = true;
  const ranges: Array<[string[], number, number, string]> = [
    [oldLines, oldStart, prefix, " "],
    [oldLines, prefix, oldChangeEnd, "-"],
    [newLines, prefix, newChangeEnd, "+"],
    [newLines, newChangeEnd, newEnd, " "],
  ];
  outer: for (const [lines, start, end, marker] of ranges) {
    for (let index = start; index < end; index++) {
      const lineBytes = 2 + Buffer.byteLength(lines[index]);
      if (fullDiffBytes + lineBytes > MAX_PERMISSION_DIFF_BYTES) {
        complete = false;
        break outer;
      }
      fullDiff.push(`${marker}${lines[index]}`);
      fullDiffBytes += lineBytes;
    }
  }
  if (complete) return fullDiff.join("\n");

  return [
    ...header,
    ` diff truncated: ${oldChangeEnd - prefix} removed, ${newChangeEnd - prefix} added lines`,
    ...sampleDiffLines(oldLines, prefix, oldChangeEnd, "-"),
    ...sampleDiffLines(newLines, prefix, newChangeEnd, "+"),
  ].join("\n");
}

function identifiedName(value: unknown): string | undefined {
  if (!value || typeof value !== "object") return undefined;
  const identifier = (value as Record<string, unknown>).identifier;
  if (!identifier || typeof identifier !== "object") return undefined;
  const text = (identifier as Record<string, unknown>).text;
  return typeof text === "string" ? text : undefined;
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function items(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function initialTitle(kind: PresentationKind, args: any): string {
  switch (kind) {
    case "read":
      return `Read ${args.path}`;
    case "view":
      return `View ${args.path}`;
    case "map":
      return `Map ${args.path}`;
    case "search":
      return `Search ${args.pattern}`;
    case "grep":
      return `Grep ${args.pattern}`;
    case "def":
      return `Find definition ${args.name}`;
    case "refs":
      return `Find references to ${args.name}`;
    case "hover":
      return `Identify ${args.path}:${args.line}`;
    case "rename":
      return `Rename ${args.path}:${args.line} to ${args.to}`;
    case "edit":
      return `Edit ${args.path}`;
    case "write":
      return `Write ${args.path}`;
    case "check":
      return `Check ${args.path}`;
  }
}

function resultPresentation(kind: PresentationKind, args: any, value: unknown): Presentation {
  const output = record(value);
  switch (kind) {
    case "read": {
      const startLine = output.start_line;
      const endLine = output.end_line;
      if (typeof startLine === "number" && typeof endLine === "number") {
        return {
          title: `Read ${args.path}:${startLine}-${endLine}`,
          metadata: { start_line: startLine, end_line: endLine, line_count: endLine - startLine + 1 },
        };
      }
      const width = output.width;
      const height = output.height;
      if (typeof width === "number" && typeof height === "number") {
        return { title: `Read ${args.path} (${width}x${height})`, metadata: { width, height } };
      }
      break;
    }
    case "view":
      return { title: `Viewed ${args.path}`, metadata: {} };
    case "map": {
      const symbols = items(output.symbols).length;
      return { title: `Mapped ${args.path} (${symbols} symbols)`, metadata: { symbols } };
    }
    case "search": {
      const results = items(output.results);
      const matches = results.reduce<number>((total, result) => total + items(record(result).matches).length, 0);
      return { title: `Found ${matches} matches`, metadata: { results: results.length, matches } };
    }
    case "grep": {
      const results = items(output.results);
      const matches = results.reduce<number>((total, result) => total + items(record(result).matches).length, 0);
      return { title: `Found ${matches} matches`, metadata: { results: results.length, matches } };
    }
    case "def": {
      const locations = items(output.locations).length;
      return { title: `Found ${locations} definitions`, metadata: { locations } };
    }
    case "refs": {
      const references = items(output.references).length;
      return { title: `Found ${references} references`, metadata: { references } };
    }
    case "hover": {
      const identifier = record(output.identifier).text;
      const line = output.line;
      const column = output.column;
      const metadata: Record<string, number> = {};
      if (typeof line === "number") metadata.line = line;
      if (typeof column === "number") metadata.column = column;
      return { title: typeof identifier === "string" ? `Identified ${identifier}` : initialTitle(kind, args), metadata };
    }
    case "rename": {
      const oldName = output.old_name;
      const newName = output.new_name;
      const otherOutputs = items(output.others).map(record);
      const edits = otherOutputs.reduce((total, item) => total + items(item.edits).length, items(output.edits).length);
      const conflicts = otherOutputs.reduce(
        (total, item) => total + items(item.conflicts).length,
        items(output.conflicts).length,
      );
      const others = otherOutputs.length;
      const title =
        typeof oldName === "string" && typeof newName === "string"
          ? `${output.applied === true ? "Renamed" : "Plan"} ${oldName} -> ${newName}`
          : initialTitle(kind, args);
      return { title, metadata: { edits, conflicts, others } };
    }
    case "edit":
    case "write":
      return { title: `${kind === "edit" ? "Edited" : "Wrote"} ${args.path}`, metadata: {} };
    case "check": {
      const errors = typeof output.error_count === "number" ? output.error_count : 0;
      const missing = typeof output.missing_count === "number" ? output.missing_count : 0;
      return {
        title: `Checked ${args.path} (${errors} errors, ${missing} missing)`,
        metadata: { error_count: errors, missing_count: missing },
      };
    }
  }
  return { title: initialTitle(kind, args), metadata: {} };
}

function readseekTool(
  description: string,
  args: Record<string, any>,
  kind: PresentationKind,
  execute: (args: any, context: ToolContext) => Promise<unknown>,
  formatResult: (result: unknown) => RenderedToolResult = (result) => ({ output: render(result) }),
) {
  return tool({
    description,
    args,
    async execute(args, context) {
      const title = initialTitle(kind, args);
      context.metadata({ title });
      const result = await execute(args, context);
      const presentation = resultPresentation(kind, args, result);
      return {
        title: presentation.title,
        ...formatResult(result),
        metadata: presentation.metadata,
      };
    },
  });
}

/**
 * Adds readseek's anchored reads and structural navigation without replacing
 * OpenCode's built-in file tools.
 */
export const ReadSeekPlugin: Plugin = async (_input, options) => {
  const anchors = new SessionAnchors();
  const imagePolicy = resolveImagePolicy(options);
  const imageModes: readonly ImageMode[] = imagePolicy === "auto"
    ? ["none", "all", "ocr", "caption", "objects"]
    : ["all", "ocr", "caption", "objects"];
  const withSearchFlags = (args: string[], input: { cached?: boolean; others?: boolean; ignored?: boolean }) => {
    if (input.ignored && !input.others) throw new Error("ignored requires others");
    optionalFlag(args, input.cached, "--cached");
    optionalFlag(args, input.others, "--others");
    optionalFlag(args, input.ignored, "--ignored");
  };

  return {
    tool: {
      readseek_read: readseekTool(
        imagePolicy === "off"
          ? "Read text with LINE:HASH anchors for durable references. Images and PDFs are skipped."
          : `Read text with LINE:HASH anchors for durable references. For images or PDFs, select a mode (${imageModes.join(", ")}) or omit image to skip.`,
        {
          path: tool.schema.string().describe("Path relative to the project directory"),
          offset: tool.schema.number().int().positive().optional().describe("One-based starting line"),
          limit: tool.schema.number().int().positive().optional().describe(`Maximum number of lines; defaults to ${DEFAULT_READ_LIMIT}`),
          page: tool.schema.number().int().positive().optional().describe("One-based PDF page; defaults to 1"),
          language: tool.schema.string().optional().describe("Language override when auto-detection is ambiguous"),
          ...(imagePolicy === "off"
            ? {}
            : {
                image: tool.schema.enum(imageModes as [ImageMode, ...ImageMode[]]).optional()
                  .describe("Image or PDF processing mode"),
              }),
        },
        "read",
        async (input, context) => {
          const filePath = resolvePath(context.directory, input.path as string);
          await authorizeRead(context, filePath);
          const image = input.image as ImageMode | undefined;
          if (imagePolicy === "off" && image !== undefined) throw new Error("image and PDF reads are disabled");
          if (imagePolicy === "on" && image === "none") throw new Error('image="none" requires imageMode="auto"');
          const detection = await runReadSeek(context, ["detect", filePath]);
          const visual = isVisualFile(detection);
          const pdf = isPdfFile(detection);
          if (visual && image === undefined) {
            return { file: filePath, skipped: true, reason: "image mode not selected" };
          }
          if (visual && (input.offset !== undefined || input.limit !== undefined)) {
            throw new Error("offset and limit do not apply to image or PDF reads");
          }
          if (visual && input.language !== undefined) {
            throw new Error("language does not apply to image or PDF reads");
          }
          if (!pdf && input.page !== undefined) throw new Error("page applies to PDF reads only");

          const args = ["read", input.offset === undefined ? filePath : `${filePath}:${input.offset}`];
          if (!visual) {
            args.push("--end", String((input.offset ?? 1) + ((input.limit as number | undefined) ?? DEFAULT_READ_LIMIT) - 1));
          }
          if (input.language) args.push("--language", input.language as string);
          if (image !== undefined) args.push("--vision-mode", image);
          if (pdf) args.push("--page", String(input.page ?? 1));
          return runReadSeek(context, args);
        },
        formatReadResult,
      ),
      readseek_view: readseekTool(
        "View the structure or selected content of an indexed PDF. Start with the overview, then narrow by page or node.",
        {
          path: tool.schema.string().describe("PDF path relative to the project directory"),
          node: tool.schema.string().min(1).optional().describe("Node ID to use as the view root"),
          page: tool.schema.number().int().positive().optional().describe("One-based source page"),
          kind: tool.schema.enum(DOCUMENT_NODE_KINDS).optional().describe("Node kind filter"),
          depth: tool.schema.number().int().nonnegative().optional().describe("Maximum depth below selected roots"),
          outline: tool.schema.boolean().optional().describe("Return outline nodes only"),
        },
        "view",
        async (input, context) => {
          const filePath = resolvePath(context.directory, input.path as string);
          await authorizeRead(context, filePath);
          const cacheDir = await ensureReadSeekCache(context);
          const args = ["--readseek-dir", cacheDir, "view", filePath];
          if (input.node) args.push("--node", input.node as string);
          if (input.page !== undefined) args.push("--page", String(input.page));
          if (input.kind) args.push("--kind", input.kind as string);
          if (input.depth !== undefined) args.push("--depth", String(input.depth));
          optionalFlag(args, input.outline as boolean | undefined, "--outline");
          return runReadSeekRaw(context, args);
        },
        (result) => ({ output: boundDocumentOutput(String(result)) }),
      ),
      readseek_map: readseekTool(
        "List symbols and ranges in a source file. Use to inspect structure without reading the full file.",
        {
          path: tool.schema.string().describe("Path relative to the project directory"),
          language: tool.schema.string().optional().describe("Language override when auto-detection is ambiguous"),
        },
        "map",
        async (input, context) => {
          const filePath = resolvePath(context.directory, input.path as string);
          await authorizeRead(context, filePath);
          const args = ["map", filePath];
          if (input.language) args.push("--language", input.language as string);
          return runReadSeek(context, args);
        },
      ),
      readseek_grep: readseekTool(
        "Search text or regular expressions and return matching LINE:HASH anchors. Use literal for exact text.",
        {
          pattern: tool.schema.string().describe("Text or regular expression to search for"),
          path: tool.schema.string().optional().describe("File or directory, defaulting to the project directory"),
          glob: tool.schema.string().optional().describe("File-name glob, such as **/*.ts"),
          literal: tool.schema.boolean().optional().describe("Treat pattern as literal text"),
          ignoreCase: tool.schema.boolean().optional().describe("Ignore case"),
          context: tool.schema.number().int().nonnegative().optional().describe("Surrounding lines to return"),
          limit: tool.schema.number().int().positive().optional().describe("Maximum matching lines; defaults to 100"),
        },
        "grep",
        async (input, context) => {
          const target = resolvePath(context.directory, (input.path as string | undefined) ?? ".");
          const pattern = input.pattern as string;
          await authorizeSearch(context, target, pattern);
          let expression: RegExp;
          try {
            const source = input.literal ? pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") : pattern;
            expression = new RegExp(source, input.ignoreCase ? "i" : "");
          } catch (error) {
            throw new Error(`invalid regular expression: ${error instanceof Error ? error.message : String(error)}`);
          }

          const info = await stat(target);
          const files: string[] = [];
          if (info.isFile()) files.push(target);
          else if (info.isDirectory()) {
            const glob = new Bun.Glob((input.glob as string | undefined) ?? "**/*");
            for await (const filePath of glob.scan({ cwd: target, absolute: true, onlyFiles: true, followSymlinks: false })) {
              files.push(filePath);
            }
          } else throw new Error(`grep target is not a file or directory: ${target}`);

          const maxMatches = (input.limit as number | undefined) ?? 100;
          const contextLines = (input.context as number | undefined) ?? 0;
          const results: { file: string; matches: unknown[]; hashlines: unknown[] }[] = [];
          let totalMatches = 0;
          for (const filePath of files.sort()) {
            context.abort.throwIfAborted();
            if (totalMatches >= maxMatches) break;
            const buffer = await readFile(filePath);
            if (buffer.includes(0)) continue;
            const lines = buffer.toString("utf8").replace(/\r\n/g, "\n").split("\n");
            const ranges: [number, number][] = [];
            for (let index = 0; index < lines.length && totalMatches < maxMatches; index++) {
              expression.lastIndex = 0;
              if (!expression.test(lines[index] ?? "")) continue;
              ranges.push([Math.max(1, index + 1 - contextLines), Math.min(lines.length, index + 1 + contextLines)]);
              totalMatches++;
            }
            if (ranges.length === 0) continue;
            const matches: unknown[] = [];
            for (const [start, end] of ranges) {
              const output = record(await runReadSeek(context, ["read", `${filePath}:${start}`, "--end", String(end)]));
              matches.push(...items(output.hashlines));
            }
            results.push({ file: filePath, matches, hashlines: matches });
          }
          return { results, truncated: totalMatches >= maxMatches };
        },
      ),
      readseek_search: readseekTool(
        "Search syntax-aware code shapes with an ast-grep pattern and return LINE:HASH anchors. Use for calls, imports, declarations, JSX, or control flow; use text search for plain text.",
        {
          pattern: tool.schema.string().describe("ast-grep pattern, such as console.log($$$ARGS)"),
          path: tool.schema.string().optional().describe("File or directory, defaulting to the project directory"),
          language: tool.schema.string().optional().describe("Language override when auto-detection is ambiguous"),
          cached: tool.schema.boolean().optional().describe("Search tracked or indexed files in a Git repository"),
          others: tool.schema.boolean().optional().describe("Search untracked files in a Git repository"),
          ignored: tool.schema.boolean().optional().describe("Include ignored untracked files; requires others"),
        },
        "search",
        async (input, context) => {
          const target = resolvePath(context.directory, (input.path as string | undefined) ?? ".");
          const args = ["search", target, input.pattern as string];
          if (input.language) args.push("--language", input.language as string);
          withSearchFlags(args, input);
          await authorizeSearch(context, target, input.pattern as string);
          const result = await runReadSeek(context, args);
          return result;
        },
      ),
      readseek_def: readseekTool(
        "Find symbol declarations by name and return LINE:HASH anchors. Use instead of text search to locate where a function, class, type, or other symbol is defined.",
        {
          name: tool.schema.string().describe("Symbol name"),
          path: tool.schema.string().optional().describe("File or directory, defaulting to the project directory"),
          language: tool.schema.string().optional().describe("Language override when auto-detection is ambiguous"),
          cached: tool.schema.boolean().optional().describe("Search tracked or indexed files in a Git repository"),
          others: tool.schema.boolean().optional().describe("Search untracked files in a Git repository"),
          ignored: tool.schema.boolean().optional().describe("Include ignored untracked files; requires others"),
        },
        "def",
        async (input, context) => {
          const target = resolvePath(context.directory, (input.path as string | undefined) ?? ".");
          const args = ["def", target, "--format", "plain", input.name as string];
          if (input.language) args.push("--language", input.language as string);
          withSearchFlags(args, input);
          await authorizeSearch(context, target, input.name as string);
          return runReadSeek(context, args);
        },
      ),
      readseek_refs: readseekTool(
        "Find identifier usages with enclosing symbols. Use before changing or deleting a symbol; add a cursor scope to exclude same-named bindings.",
        {
          name: tool.schema.string().describe("Identifier name"),
          path: tool.schema.string().optional().describe("File or directory, defaulting to the project directory"),
          language: tool.schema.string().optional().describe("Language override when auto-detection is ambiguous"),
          scope: tool.schema.boolean().optional().describe("Restrict results to the binding at the cursor"),
          line: tool.schema.number().int().positive().optional().describe("Cursor line, required with scope"),
          column: tool.schema.number().int().positive().optional().describe("Cursor byte column for disambiguation"),
          cached: tool.schema.boolean().optional().describe("Search tracked or indexed files in a Git repository"),
          others: tool.schema.boolean().optional().describe("Search untracked files in a Git repository"),
          ignored: tool.schema.boolean().optional().describe("Include ignored untracked files; requires others"),
        },
        "refs",
        async (input, context) => {
          if (input.scope && input.line === undefined) throw new Error("scope requires line");
          if (!input.scope && (input.line !== undefined || input.column !== undefined)) {
            throw new Error("line and column require scope");
          }
          const target = resolvePath(context.directory, (input.path as string | undefined) ?? ".");
          const args = ["refs", target, input.name as string];
          optionalFlag(args, input.scope as boolean | undefined, "--scope");
          if (input.line !== undefined) args.push("--line", String(input.line));
          if (input.column !== undefined) args.push("--column", String(input.column));
          if (input.language) args.push("--language", input.language as string);
          withSearchFlags(args, input);
          await authorizeSearch(context, target, input.name as string);
          return runReadSeek(context, args);
        },
      ),
      readseek_hover: readseekTool(
        "Identify the token and enclosing symbol at a cursor. Use before definition lookup or rename, or to identify a line's enclosing symbol.",
        {
          path: tool.schema.string().describe("Path relative to the project directory"),
          line: tool.schema.number().int().positive().describe("One-based cursor line"),
          column: tool.schema.number().int().positive().optional().describe("One-based cursor byte column"),
          language: tool.schema.string().optional().describe("Language override when auto-detection is ambiguous"),
        },
        "hover",
        async (input, context) => {
          const filePath = resolvePath(context.directory, input.path as string);
          await authorizeRead(context, filePath);
          const args = ["identify", `${filePath}:${input.line}`];
          if (input.column !== undefined) args.push("--column", String(input.column));
          if (input.language) args.push("--language", input.language as string);
          return runReadSeek(context, args);
        },
      ),
      readseek_rename: readseekTool(
        "Rename a symbol without changing same-named bindings. Applies verified edits atomically by default; set apply false for a dry-run plan.",
        {
          path: tool.schema.string().describe("Path relative to the project directory"),
          line: tool.schema.number().int().positive().describe("One-based line of the symbol to rename"),
          column: tool.schema.number().int().positive().optional().describe("One-based byte column for disambiguation"),
          to: tool.schema.string().min(1).describe("New symbol name"),
          workspace: tool.schema.boolean().optional().describe("Include project-wide occurrences"),
          apply: tool.schema.boolean().optional().describe("Apply verified edits atomically; defaults to true"),
        },
        "rename",
        async (input, context) => {
          const filePath = resolvePath(context.directory, input.path as string);
          await authorizeRead(context, filePath);
          if (input.workspace) {
            const identifyArgs = ["identify", `${filePath}:${input.line}`];
            if (input.column !== undefined) identifyArgs.push("--column", String(input.column));
            const name = identifiedName(await runReadSeek(context, identifyArgs));
            if (!name) throw new Error("readseek could not identify a binding at the rename cursor");
            await authorizeSearch(context, context.directory, name);
          }
          const args = ["rename", filePath, "--line", String(input.line), "--to", input.to as string];
          if (input.column !== undefined) args.push("--column", String(input.column));
          if (input.workspace) args.push("--workspace", context.directory);
          const plan = await runReadSeek(context, args);
          if (input.apply === false) return plan;
          const files = renameFiles(plan);
          const planHash = record(plan).plan_hash;
          if (typeof planHash !== "string" || planHash.length === 0) {
            throw new Error("readseek rename plan did not include a plan hash");
          }
          await authorizeEdit(context, files);
          return runReadSeek(context, [...args, "--plan-hash", planHash, "--apply"], { cancelable: false });
        },
      ),
      readseek_edit: readseekTool(
        "Edit an existing text file using LINE:HASH anchors. Stale hashes are rejected before writing.",
        {
          path: tool.schema.string().describe("Path relative to the project directory"),
          edits: tool.schema.array(tool.schema.union([
            tool.schema.object({
              set_line: tool.schema.object({ anchor: tool.schema.string(), new_text: tool.schema.string() }).strict(),
            }).strict(),
            tool.schema.object({
              replace_lines: tool.schema.object({
                start_anchor: tool.schema.string(),
                end_anchor: tool.schema.string(),
                new_text: tool.schema.string(),
              }).strict(),
            }).strict(),
            tool.schema.object({
              insert_after: tool.schema.object({ anchor: tool.schema.string(), new_text: tool.schema.string() }).strict(),
            }).strict(),
          ])).min(1).describe("Anchored edits to apply"),
        },
        "edit",
        async (input, context) => {
          const filePath = resolvePath(context.directory, input.path as string);
          validateAnchorEdits(input.edits);
          return withFileMutationQueue(filePath, async () => {
            context.abort.throwIfAborted();
            await authorizeRead(context, filePath);
            await rejectSymlinkMutation(filePath);
            const edits = input.edits as AnchorEdit[];
            const before = await readFile(filePath, "utf8");
            await verifyAnchors(context, filePath, edits);
            const after = applyAnchorEdits(before, edits);
            await authorizeEdit(context, [filePath], simpleDiff(filePath, before, after), false);
            context.abort.throwIfAborted();
            if (before !== after) await writeText(filePath, { exists: true, content: before }, after);
            const firstEditedLine = Math.min(...edits.flatMap((edit) => anchorRefs(edit).map((ref) => ref.line)));
            const previewStart = Math.max(1, firstEditedLine - 3);
            return runReadSeek(
              context,
              ["read", `${filePath}:${previewStart}`, "--end", String(previewStart + EDIT_RESULT_READ_LIMIT - 1)],
              { cancelable: false },
            );
          });
        },
      ),
      readseek_write: readseekTool(
        "Create or replace a whole text file and return LINE:HASH anchors.",
        {
          path: tool.schema.string().describe("Path relative to the project directory"),
          content: tool.schema.string().describe("Complete text file content"),
        },
        "write",
        async (input, context) => {
          const filePath = resolvePath(context.directory, input.path as string);
          await authorizeExternal(context, filePath);
          return withFileMutationQueue(filePath, async () => {
            context.abort.throwIfAborted();
            await rejectSymlinkMutation(filePath);
            const before = await readTextSnapshot(filePath);
            const content = input.content as string;
            if (content.includes("\0")) throw new Error("write content must be text");
            await authorizeEdit(context, [filePath], simpleDiff(filePath, before.content, content), false);
            context.abort.throwIfAborted();
            await writeText(filePath, before, content);
            return runReadSeek(context, ["read", filePath, "--end", String(DEFAULT_READ_LIMIT)], { cancelable: false });
          });
        },
      ),
      readseek_check: readseekTool(
        "Check a source file for parser errors and missing syntax. Use after edits for quick syntax validation, not as a compiler or linter.",
        {
          path: tool.schema.string().describe("Path relative to the project directory"),
          language: tool.schema.string().optional().describe("Language override when auto-detection is ambiguous"),
        },
        "check",
        async (input, context) => {
          const filePath = resolvePath(context.directory, input.path as string);
          await authorizeRead(context, filePath);
          const args = ["check", filePath];
          if (input.language) args.push("--language", input.language as string);
          return runReadSeek(context, args);
        },
      ),
    },
    "experimental.chat.system.transform": async (_input, output) => {
      output.system.push(TOOL_POLICY);
    },
    "tool.definition": async (input, output) => {
      const preference = PREFERRED_TOOL_DESCRIPTIONS[input.toolID];
      if (preference) {
        output.description = `${preference} ${output.description}`;
        return;
      }
      const fallback = FALLBACK_TOOL_DESCRIPTIONS[input.toolID];
      if (fallback) output.description = `${fallback} ${output.description}`;
    },
    event: async ({ event }) => {
      if (event.type === "file.edited" || event.type === "file.watcher.updated") {
        anchors.forget(event.properties.file);
        return;
      }
      if (event.type === "session.deleted") anchors.deleteSession(event.properties.info.id);
    },
    "tool.execute.after": async (input, output) => {
      if (!input.tool.startsWith("readseek_")) return;
      try {
        const result = JSON.parse(output.output) as unknown;
        if (input.tool === "readseek_rename") {
          anchors.planRename(input.sessionID, result);
          if (record(result).applied === true) {
            for (const filePath of renameFiles(result)) anchors.forget(filePath);
          }
        }

        const files = new Set<string>();
        collectAnchoredFiles(result, files);
        for (const filePath of files) anchors.mark(input.sessionID, path.resolve(filePath));
      } catch {
        // A failed tool result has no valid anchors to retain.
      }
    },
    "experimental.session.compacting": async (input, output) => {
      const context = anchors.render(input.sessionID);
      if (context) output.context.push(context);
    },
  };
};

export default { id: "opencode-readseek", server: ReadSeekPlugin };
