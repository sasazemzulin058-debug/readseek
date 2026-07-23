import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, relative } from "node:path";

import { withFileMutationQueue, truncateHead, type ExtensionAPI, type ToolRenderResultOptions } from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";
import { Type } from "@sinclair/typebox";

import { resolveToCwd } from "./path-utils.js";
import { ensureHashInit, formatHashlineDisplay } from "./hashline.js";
import { buildReadSeekLine, buildReadSeekWarning, buildToolErrorResult, type ReadSeekLine, type ReadSeekWarning, type ToolErrorResult } from "./readseek-value.js";
import { looksLikeBinary } from "./binary-detect.js";
import { formatFsError } from "./fs-error.js";
import { getOrGenerateMap } from "./file-map.js";
import { formatFileMapWithBudget } from "./readseek/formatter.js";

import { defineToolPromptMetadata } from "./tool-prompt-metadata.js";
import { buildPendingWritePreviewData, buildWritePreviewKey, resolvePendingDiffPreview, type PendingDiffPreviewResult } from "./pending-diff-preview.js";
import { generateCompactOrFullDiff, normalizeToLF, hasBareCarriageReturn } from "./edit-diff.js";
import { buildDiffData, type DiffData } from "./diff-data.js";
import { clampLineToWidth, clampLinesToWidth, linkToolPath, renderErrorResult, renderPendingResult, renderToolLabel, resolveRenderResultContext, summaryLine } from "./tui-render-utils.js";
import { resolveReadSeekToolDisplayMode } from "./readseek-settings.js";
import { upsertDiffComponent, upsertTextComponent } from "./tui-diff-component.js";
import type { FileAnchoredCallback } from "./tool-types.js";
import { filePathParam, mapParam, registerReadSeekTool } from "./register-tool.js";

const WRITE_PENDING_PREVIEW_STATE_KEY = "hashline-write-pending-preview";

const CONTENT_PREVIEW_MAX_LINES = 200;

function formatContentPreviewLines(content: string, theme: any): string[] {
  const lines = content.split("\n");
  // Drop the single trailing blank produced by a terminal newline so the
  // preview reads naturally.
  if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
  const shown = lines.slice(0, CONTENT_PREVIEW_MAX_LINES);
  // Right-align line numbers so the body has a stable column for the content.
  // The dim gutter ("  N │ ") visually distinguishes the body from the
  // "↳ created / pending create" header above it without re-introducing
  // diff chrome (no +/- marker, no red/green tint).
  const width = String(shown.length).length;
  // Bind theme.fg so we keep its `this` (the Theme uses internal state); fall back
  // to an identity tint when no theme is provided (e.g. in tests).
  const fg = typeof theme?.fg === "function" ? (style: string, text: string) => theme.fg(style, text) : (_style: string, text: string) => text;
  const formatted = shown.map((line, index) => {
    const gutter = fg("dim", `${String(index + 1).padStart(width, " ")} │ `);
    return `  ${gutter}${line}`;
  });
  if (lines.length > CONTENT_PREVIEW_MAX_LINES) {
    formatted.push(`  ${fg("dim", `… ${lines.length - CONTENT_PREVIEW_MAX_LINES} more lines not shown`)}`);
  }
  return formatted;
}

function pendingWritePreviewParts(summary: string, preview: PendingDiffPreviewResult | undefined, expanded: boolean, theme: any): { lines: string[]; diffData?: DiffData } {
  if (!preview || preview.type !== "ok") return { lines: summary.split("\n") };
  // Pure creates (write to a new file) have no "old" side, so a diff-shaped
  // preview is just noise. Show the new file's content with a dim gutter of
  // line numbers when expanded; otherwise just a Ctrl+O hint.
  const hasOldSide = preview.data.fileExistedBeforeWrite;
  const headerLine = summaryLine(preview.data.headerLabel, { hidden: !expanded });
  if (!hasOldSide) {
    const lines = [summary, headerLine];
    if (expanded) lines.push(...formatContentPreviewLines(preview.data.nextContent, theme));
    return { lines };
  }
  const diffData = buildDiffData({ path: preview.data.filePath, oldContent: preview.data.previousContent, newContent: preview.data.nextContent, diff: preview.data.diff });
  return { lines: [summary, headerLine], diffData: expanded ? diffData : undefined };
}


type WriteDiffFields = {
  diff?: string;
  diffData?: DiffData;
};

export interface WriteResult extends WriteDiffFields {
  text: string;
  warnings: string[];
  writeState?: "created" | "overwritten";
  readSeekValue: {
    tool: "write";
    path: string;
    lines: ReadSeekLine[];
    warnings: ReadSeekWarning[];
    diff?: string;
    diffData?: DiffData;
    map?: { appended: boolean };
  };
}


type ExecuteWriteResult = WriteResult | ToolErrorResult;

function isToolErrorResult(result: ExecuteWriteResult): result is ToolErrorResult {
  return "isError" in result && result.isError === true;
}

async function readPreviousTextForDiff(filePath: string): Promise<{ content: string; existed: boolean }> {
  try {
    const previous = await readFile(filePath);
    return {
      content: looksLikeBinary(previous) ? "" : previous.toString("utf-8"),
      existed: true,
    };
  } catch (err: unknown) {
    if ((err as { code?: unknown } | null)?.code === "ENOENT") {
      return { content: "", existed: false };
    }
    throw err;
  }
}

function generateWriteDiff(previousContent: string, nextContent: string): { diff: string; firstChangedLine: number | undefined } {
  if (previousContent !== "") return generateCompactOrFullDiff(previousContent, nextContent);
  const normalizedNext = normalizeToLF(nextContent);
  if (normalizedNext === "") return { diff: "", firstChangedLine: undefined };
  const lines = normalizedNext.split("\n");
  if (lines[lines.length - 1] === "") lines.pop();
  const width = String(lines.length).length;
  return {
    diff: lines.map((line, index) => `+${String(index + 1).padStart(width, " ")} ${line}`).join("\n"),
    firstChangedLine: 1,
  };
}

export interface WriteToolOptions {
  onFileAnchored?: FileAnchoredCallback;
  name?: string;
}

function buildWriteFsErrorResult(err: any, absolutePath: string) {
	const { code, message } = formatFsError(err, "write-error");
	return buildToolErrorResult("write", code, message, {
		path: absolutePath,
		extra: { lines: [], warnings: [] },
	});
}

export async function executeWrite(opts: {
  path: string;
  content: string;
  map?: boolean;
  cwd?: string;
}): Promise<ExecuteWriteResult> {
  await ensureHashInit();

  const { path: filePath, content, map: requestMap, cwd } = opts;
  const warnings: string[] = [];
  const readSeekWarnings: ReadSeekWarning[] = [];

  if (hasBareCarriageReturn(content)) {
    const message = "File content contains bare CR (\\r) line endings; readSeek_write refuses to emit anchors that readSeek_read/readSeek_edit would normalize differently.";
    warnings.push(message);
    readSeekWarnings.push(buildReadSeekWarning("bare-cr", message));
    return buildToolErrorResult("write", "bare-cr", `Cannot write ${filePath}\n⚠️ ${message}`, {
      path: filePath,
      extra: { lines: [], warnings: readSeekWarnings },
    });
  }
  if (looksLikeBinary(Buffer.from(content, "utf-8"))) {
    const message = "File content appears to be binary.";
    warnings.push(message);
    readSeekWarnings.push(buildReadSeekWarning("binary-content", message));
    return buildToolErrorResult("write", "binary-content", `Cannot write ${filePath}\n⚠️ ${message} — refusing to write.`, {
      path: filePath,
      extra: { lines: [], warnings: readSeekWarnings },
    });
  }

  const { content: previousContent, existed: existedBeforeWrite } = await readPreviousTextForDiff(filePath);

	// Create parent directories
	await mkdir(dirname(filePath), { recursive: true });
	// Write file
	await writeFile(filePath, content, "utf-8");
  // Compute hashlines
  const rawLines = content.split("\n");
  const readSeekLines: ReadSeekLine[] = [];
  const displayLines: string[] = [];

  for (let i = 0; i < rawLines.length; i++) {
    const lineNum = i + 1;
    const readSeekLine = buildReadSeekLine(lineNum, rawLines[i]);
    readSeekLines.push(readSeekLine);
    displayLines.push(formatHashlineDisplay(lineNum, rawLines[i]));
  }

  const fullText = displayLines.join("\n");
  const truncated = truncateHead(fullText);
  let text = truncated.content;
  if (truncated.truncated) {
    if (truncated.truncatedBy === "lines") {
      text += `\n[… ${truncated.totalLines - truncated.outputLines} more lines not shown — full anchors in readSeekValue]`;
    } else {
      text += "\n[… output truncated at 50 KB — full anchors in readSeekValue]";
    }
  }

  // Optional structural map
  let mapAppended = false;
  if (requestMap) {
    try {
      const fileMap = await getOrGenerateMap(filePath);
      if (fileMap) {
        const mapText = formatFileMapWithBudget(fileMap);
        if (mapText) {
          text += "\n\n" + mapText;
          mapAppended = true;
        }
      }
    } catch {
      // Map generation failure is non-fatal
    }
  }

  const displayPath = cwd ? relative(cwd, filePath) || filePath : filePath;
  const normalizedPrevious = normalizeToLF(previousContent);
  const normalizedNext = normalizeToLF(content);
  const diffResult = generateWriteDiff(normalizedPrevious, normalizedNext);
  const diffData = buildDiffData({
    path: filePath,
    oldContent: normalizedPrevious,
    newContent: normalizedNext,
    diff: diffResult.diff,
  });

  return {
    text,
    warnings,
    writeState: existedBeforeWrite ? "overwritten" : "created",
    diff: diffResult.diff,
    diffData,
    readSeekValue: {
      tool: "write",
      path: displayPath,
      lines: readSeekLines,
      warnings: readSeekWarnings,
      diff: diffResult.diff,
      diffData,
      ...(requestMap ? { map: { appended: mapAppended } } : {}),
    },
  };
}

export function registerWriteTool(pi: ExtensionAPI, options: WriteToolOptions = {}) {
  const name = options.name ?? "readSeek_write";
  const promptMetadata = defineToolPromptMetadata({
    promptUrl: new URL("../prompts/write.md", import.meta.url),
    promptSnippet: "Create or replace a complete file with edit anchors",
    registeredName: name,
  });
  const tool = registerReadSeekTool(pi, {
    name,
    label: "write",
    description: promptMetadata.description,
    promptSnippet: promptMetadata.promptSnippet,
    promptGuidelines: promptMetadata.promptGuidelines,
    parameters: Type.Object({
      path: filePathParam(),
      content: Type.String({ description: "Complete file content" }),
      map: mapParam(),
    }),
    async execute(_toolCallId: string, params: { path: string; content: string; map?: boolean }, _signal: AbortSignal | undefined, _onUpdate: any, ctx: any): Promise<any> {
      const cwd = ctx?.cwd ?? process.cwd();
      const absolutePath = resolveToCwd(params.path, cwd);
      try {
        return await withFileMutationQueue(absolutePath, async () => {
          let result: ExecuteWriteResult;
          try {
            result = await executeWrite({
              path: absolutePath,
              content: params.content,
              map: params.map,
              cwd,
            });
          } catch (err: any) {
            return buildWriteFsErrorResult(err, absolutePath);
          }

          if (isToolErrorResult(result)) return result;

          if (result.readSeekValue.lines.length > 0) {
            options.onFileAnchored?.(absolutePath);
          }

          return {
            content: [{ type: "text" as const, text: result.text }],
            details: {
              ...(result.diff !== undefined ? { diff: result.diff } : {}),
              ...(result.diffData !== undefined ? { diffData: result.diffData } : {}),
              ...(result.writeState ? { writeState: result.writeState } : {}),
              readSeekValue: result.readSeekValue,
              warnings: result.warnings,
            },
          };
        });
      } catch (err: any) {
        return buildWriteFsErrorResult(err, absolutePath);
      }
    },
    renderCall(args: any, theme: any, context: any = {}) {
      const { path, content } = args as { path: string; content?: string };
      const cwd = context.cwd ?? process.cwd();
      const label = renderToolLabel(theme, "write");
      const lineCount = typeof content === "string" ? content.split("\n").length : 0;
      const bytes = typeof content === "string" ? Buffer.byteLength(content, "utf8") : 0;
      const renderedPath = typeof path === "string"
        ? linkToolPath(theme.fg("muted", path), path, cwd)
        : theme.fg("toolOutput", "...");
      let text = clampLineToWidth(`${label} ${renderedPath}${typeof content === "string" ? ` (${lineCount} ${lineCount === 1 ? "line" : "lines"} • ${bytes} B)` : ""}`, context.width);
      // Once execution has started, the pending preview's only job is done:
      // renderResult will carry the story ("↳ created" / "↳ overwritten" with
      // expandable content or diff). Showing the "↳ pending…" sub-line and
      // its preview alongside the final result is just duplicate noise — the
      // pre-execution state can no longer change in any meaningful way.
      if (context.executionStarted) {
        return upsertTextComponent(context.lastComponent, text);
      }
      const previewKey = buildWritePreviewKey(args ?? {});
      const preview = resolvePendingDiffPreview(context, WRITE_PENDING_PREVIEW_STATE_KEY, previewKey, () => buildPendingWritePreviewData(args ?? {}, context.cwd ?? process.cwd()));
      const expanded = !!context.expanded;
      const parts = pendingWritePreviewParts(text, preview, expanded, theme);
      if (parts.diffData) {
        return upsertDiffComponent(context.lastComponent, { prefixLines: parts.lines, diffData: parts.diffData, theme, expanded: true });
      }
      return upsertTextComponent(context.lastComponent, clampLinesToWidth(parts.lines, context.width).join("\n"));
    },
    renderResult(result: any, options: ToolRenderResultOptions, theme: any, ...rest: any[]) {
      const { isPartial, expanded, width, context } = resolveRenderResultContext(
        options,
        rest,
        resolveReadSeekToolDisplayMode("write") === "expanded",
      );
      if (isPartial) return renderPendingResult("pending write", width, theme);
      const details = result.details ?? {};
      const output = result.content?.[0]?.type === "text" ? result.content[0].text : "";
      if (result.isError || details.readSeekValue?.ok === false) {
        return renderErrorResult(output, { expanded, width, fallback: "write failed", theme });
      }
      const diffData = details.diffData;
      const state = details.writeState === "overwritten" ? "overwritten" : "created";
      // Pure creates: render the new file's contents on expand (no diff chrome)
      // instead of a diff body — every line is an add, so the gutter, line
      // numbers, and red/green tinting are noise.
      if (state === "created") {
        const readSeekLines = (details.readSeekValue?.lines ?? []) as Array<{ raw: string }>;
        const hasContent = readSeekLines.length > 0;
        const header = summaryLine(state, { hidden: hasContent && !expanded, theme, style: "success" });
        const lines = header.split("\n");
        if (expanded && hasContent) {
          const content = readSeekLines.map((l) => l.raw).join("\n");
          lines.push(...formatContentPreviewLines(content, theme));
        }
        return new Text(clampLinesToWidth(lines, width).join("\n"), 0, 0);
      }
      // Overwrite: the old vs new comparison still carries signal — keep the diff UI.
      const hasExpandableDiff = !!diffData;
      let text = summaryLine(state, { hidden: hasExpandableDiff && !expanded, theme, style: "success" });
      if (expanded && hasExpandableDiff) {
        return upsertDiffComponent(context.lastComponent, { prefixLines: text.split("\n"), diffData, theme, expanded: true });
      }
      return new Text(clampLinesToWidth(text.split("\n"), width).join("\n"), 0, 0);
    },
  });
  return tool;
}
