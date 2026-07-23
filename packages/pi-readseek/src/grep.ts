import { readFile as fsReadFile, stat as fsStat } from "node:fs/promises";
import path from "node:path";

import type { ExtensionAPI, ToolRenderResultOptions } from "@earendil-works/pi-coding-agent";
import { createGrepTool } from "@earendil-works/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import { Text } from "@earendil-works/pi-tui";

import { defineToolPromptMetadata } from "./tool-prompt-metadata.js";
import { normalizeToLF, stripBom, hasBareCarriageReturn } from "./edit-diff.js";
import { looksLikeBinary } from "./binary-detect.js";
import { ensureHashInit, escapeControlCharsForDisplay } from "./hashline.js";
import { buildReadSeekLine, buildToolErrorResult, type ReadSeekLine } from "./readseek-value.js";
import { buildGrepOutput, type GrepOutputEntry, type GrepOutputGroup, type GrepOutputRecord, type GrepScopeWarning } from "./grep-output.js";

import { getOrGenerateMap } from "./file-map.js";
import { scopeGrepGroupsToSymbols } from "./grep-symbol-scope.js";
import { resolveToCwd } from "./path-utils.js";
import { throwIfAborted } from "./runtime.js";
import { formatGrepCallText, formatGrepResultText, GREP_TRUNCATION_THRESHOLD } from "./grep-render-helpers.js";
import { coerceObviousBase10Int } from "./coerce-obvious-int.js";
import { clampLineToWidth, clampLinesToWidth, linkToolPath, renderErrorResult, renderPendingResult, renderToolLabel, resolveRenderResultContext, summaryLine } from "./tui-render-utils.js";
import { renderGrepSourceForDisplay } from "./tui-source-render.js";
import type { FileAnchoredCallback } from "./tool-types.js";
import { optionalIntOrString, registerReadSeekTool } from "./register-tool.js";
import { searchPathParam } from "./readseek-params.js";


const grepSchema = Type.Object({
	pattern: Type.String({ description: "Regex pattern; use literal for exact text" }),
	path: searchPathParam(),
	glob: Type.Optional(Type.String({ description: "File-name glob, such as *.ts" })),
	ignoreCase: Type.Optional(Type.Boolean({ description: "Ignore case" })),
	literal: Type.Optional(Type.Boolean({ description: "Treat pattern literally" })),
	context: optionalIntOrString("Surrounding lines for each match"),
	limit: optionalIntOrString("Maximum matches to return"),
	summary: Type.Optional(Type.Boolean({ description: "Return per-file counts" })),
	scope: Type.Optional(
		Type.Literal("symbol", {
			description: "Group matches by enclosing symbol",
		}),
	),
	scopeContext: optionalIntOrString("Context lines within each symbol"),
});

interface GrepParams {
	pattern: string;
	path?: string;
	glob?: string;
	ignoreCase?: boolean;
	literal?: boolean;
	context?: number | string;
	limit?: number | string;
	summary?: boolean;
	scope?: "symbol";
	scopeContext?: number | string;
}

const MATCH_LINE_RE = /^(.*?):(\d+): (.*)$/;
const CONTEXT_LINE_RE = /^(.*?)-(\d+)- (.*)$/;

function parseGrepOutputLine(line: string):
	| { kind: "match"; displayPath: string; lineNumber: number; text: string }
	| { kind: "context"; displayPath: string; lineNumber: number; text: string }
	| null {
	const match = line.match(MATCH_LINE_RE);
	if (match) {
		return {
			kind: "match",
			displayPath: match[1],
			lineNumber: Number.parseInt(match[2], 10),
			text: match[3],
		};
	}

	const context = line.match(CONTEXT_LINE_RE);
	if (context) {
		return {
			kind: "context",
			displayPath: context[1],
			lineNumber: Number.parseInt(context[2], 10),
			text: context[3],
		};
	}

	return null;
}

const GREP_MAX_MATCHES_PER_FILE = 10;

type GrepAnchoredEntry = Extract<GrepOutputEntry, { kind: "match" | "context" }>;

/**
 * Collapse context lines that duplicate a match line by line number (a match
 * supersedes a context entry at the same line), then sort ascending and insert
 * a `--` separator across each line-number gap.
 */
function dedupeContextEntries(entries: GrepOutputEntry[]): GrepOutputEntry[] {
	if (entries.length === 0) return entries;

	const byLine = new Map<number, GrepAnchoredEntry>();
	for (const entry of entries) {
		if (entry.kind === "separator") continue;
		const existing = byLine.get(entry.line.line);
		if (!existing || (entry.kind === "match" && existing.kind === "context")) {
			byLine.set(entry.line.line, entry);
		}
	}

	const sorted = [...byLine.entries()].sort(([a], [b]) => a - b);
	const result: GrepOutputEntry[] = [];
	for (let i = 0; i < sorted.length; i++) {
		if (i > 0 && sorted[i][0] > sorted[i - 1][0] + 1) {
			result.push({ kind: "separator", text: "--" });
		}
		result.push(sorted[i][1]);
	}

	return result;
}

/**
 * Keep at most {@link GREP_MAX_MATCHES_PER_FILE} matches (with their context)
 * per file, appending a `... +N more matches` separator when matches are
 * dropped. The group's `matchCount` retains the pre-truncation total.
 */
function truncateGroupEntries(group: GrepOutputGroup): GrepOutputGroup {
	let matchesSeen = 0;
	let truncatedCount = 0;
	const kept: GrepOutputEntry[] = [];

	for (const entry of group.entries) {
		if (entry.kind === "match") {
			matchesSeen++;
			if (matchesSeen <= GREP_MAX_MATCHES_PER_FILE) {
				kept.push(entry);
			} else {
				truncatedCount++;
			}
		} else if (matchesSeen <= GREP_MAX_MATCHES_PER_FILE) {
			kept.push(entry);
		}
	}

	if (truncatedCount > 0) {
		kept.push({ kind: "separator", text: `... +${truncatedCount} more matches` });
	}

	return { ...group, entries: kept };
}

function recordsFromGroups(groups: GrepOutputGroup[]): GrepOutputRecord[] {
	return groups.flatMap((group) =>
		group.entries.flatMap((entry) =>
			entry.kind === "separator" ? [] : [{ path: group.absolutePath, ...entry.line, kind: entry.kind }],
		),
	);
}

/**
 * Escape special regex characters in a literal string for use in `new RegExp()`.
 */
function escapeForRegex(s: string): string {
	return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

interface GrepToolOptions {
	onFileAnchored?: FileAnchoredCallback;
	name?: string;
}

export interface ExecuteGrepOptions {
	toolCallId: string;
	params: unknown;
	signal: AbortSignal | undefined;
	onUpdate: any;
	cwd: string;
	onFileAnchored?: FileAnchoredCallback;
}

export async function executeGrep(opts: ExecuteGrepOptions): Promise<any> {
	const { toolCallId, params, signal, onUpdate, cwd, onFileAnchored } = opts;
	await ensureHashInit();
	const rawParams = params as GrepParams;
	const context = coerceObviousBase10Int(rawParams.context, "context");
	if (!context.ok) {
		return buildToolErrorResult("grep", "invalid-params-combo", context.message);
	}
	const limit = coerceObviousBase10Int(rawParams.limit, "limit");
	if (!limit.ok) {
		return buildToolErrorResult("grep", "invalid-limit", limit.message);
	}
	const scopeContext = coerceObviousBase10Int(rawParams.scopeContext, "scopeContext");
	if (!scopeContext.ok) {
		return buildToolErrorResult("grep", "invalid-params-combo", scopeContext.message);
	}
	if (scopeContext.value !== undefined && rawParams.scope !== "symbol") {
		const message = 'Invalid scopeContext: requires scope: "symbol". For normal surrounding-line context outside symbol scope, use the `context` parameter.';
		return buildToolErrorResult("grep", "invalid-params-combo", message);
	}
	if (scopeContext.value !== undefined && scopeContext.value < 0) {
		const message = `Invalid scopeContext: expected a non-negative integer, received ${scopeContext.value}.`;
		return buildToolErrorResult("grep", "invalid-params-combo", message);
	}
	const p: GrepParams = {
		...rawParams,
		context: context.value,
		limit: limit.value,
		scopeContext: scopeContext.value,
	};
	const builtin = createGrepTool(cwd);
	const result = await builtin.execute(
		toolCallId,
		{
			...p,
			context: context.value,
			limit: limit.value,
		},
		signal,
		onUpdate,
	);

	const textBlock = result.content?.find(
		(item): item is { type: "text"; text: string } =>
			item.type === "text" && "text" in item && typeof (item as { text?: unknown }).text === "string",
	);
	if (!textBlock?.text) return result;

	const { path: rawSearchPath } = p;
	const searchPath = resolveToCwd(rawSearchPath || ".", cwd);

	let searchPathIsDirectory = false;
	try {
		searchPathIsDirectory = (await fsStat(searchPath)).isDirectory();
	} catch {
		searchPathIsDirectory = false;
	}
	// Warn when the user targets a single binary file directly — grep
	// silently skips binary files and would return 0 matches with no
	// indication of why.
	if (!searchPathIsDirectory) {
		try {
			const buf = await fsReadFile(searchPath);
			if (looksLikeBinary(buf)) {
				const warning = `[Warning: '${p.path ?? searchPath}' appears to be a binary file — grep skips binary files by default. Use a hex tool or the read tool to inspect it.]`;
				return {
					...result,
					content: result.content.map((item) =>
						item === textBlock ? ({ ...item, text: warning } as typeof item) : item,
					),
					details: {
						...(typeof result.details === "object" && result.details !== null ? result.details : {}),
						readSeekValue: {
							tool: "grep",
							summary: !!p.summary,
							totalMatches: 0,
							records: [],
						},
					},
				};
			}
		} catch {
			// can't read file — let normal flow continue
		}
	}

	const fileCache = new Map<string, string[] | undefined>();
	const bareCRFiles = new Set<string>();
	const getFileLines = async (absolutePath: string): Promise<string[] | undefined> => {
		throwIfAborted(signal);
		if (fileCache.has(absolutePath)) return fileCache.get(absolutePath);
		try {
			const rawBuffer = await fsReadFile(absolutePath);
			if (looksLikeBinary(rawBuffer)) {
				fileCache.set(absolutePath, undefined);
				return undefined;
			}
			const raw = rawBuffer.toString("utf-8");
			if (hasBareCarriageReturn(raw)) bareCRFiles.add(absolutePath);
			const lines = normalizeToLF(stripBom(raw).text).split("\n");
			fileCache.set(absolutePath, lines);
			return lines;
		} catch {
			fileCache.set(absolutePath, undefined);
			return undefined;
		}
	};

	const toAbsolutePath = (displayPath: string): string => {
		if (searchPathIsDirectory) return path.resolve(searchPath, displayPath);
		return searchPath;
	};

	const groupsByPath = new Map<string, GrepOutputGroup>();
	const passthroughLines: string[] = [];
	let totalMatches = 0;
	let parsedCount = 0;
	let candidateUnparsedCount = 0;
	const candidateLinePattern = /^.+(?::|-)\d+(?::|-)\s/;
	// Pre-read all matched files with bounded concurrency so the processing
	// loop below hits the cache on every getFileLines call instead of
	// serialising disk I/O.
	if (!p.summary) {
		const pathsToRead = new Set<string>();
		for (const line of textBlock.text.split("\n")) {
			throwIfAborted(signal);
			const parsed = parseGrepOutputLine(line);
			if (parsed && Number.isFinite(parsed.lineNumber) && parsed.lineNumber >= 1) {
				pathsToRead.add(toAbsolutePath(parsed.displayPath));
			}
		}
		const CONCURRENCY = 8;
		const pathList = [...pathsToRead];
		for (let i = 0; i < pathList.length; i += CONCURRENCY) {
			await Promise.all(pathList.slice(i, i + CONCURRENCY).map((p) => getFileLines(p)));
		}
	}

	const addSummaryMatch = (displayPath: string, absolutePath: string) => {
		let group = groupsByPath.get(displayPath);
		if (!group) {
			group = { displayPath, absolutePath, matchCount: 0, entries: [] };
			groupsByPath.set(displayPath, group);
		}
		group.matchCount++;
		totalMatches++;
	};

	const addEntry = (displayPath: string, absolutePath: string, kind: "match" | "context", line: ReadSeekLine) => {
		let group = groupsByPath.get(displayPath);
		if (!group) {
			group = { displayPath, absolutePath, matchCount: 0, entries: [] };
			groupsByPath.set(displayPath, group);
		}
		group.entries.push({ kind, line });
		if (kind === "match") {
			group.matchCount++;
			totalMatches++;
		}
	};

	for (const line of textBlock.text.split("\n")) {
		throwIfAborted(signal);
		const parsed = parseGrepOutputLine(line);
		if (!parsed || !Number.isFinite(parsed.lineNumber) || parsed.lineNumber < 1) {
			if (candidateLinePattern.test(line)) {
				candidateUnparsedCount++;
			}
			const trimmed = line.trim();
			if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
				passthroughLines.push(trimmed);
			}
			continue;
		}
		parsedCount++;
		const absolute = toAbsolutePath(parsed.displayPath);
		if (p.summary) {
			if (parsed.kind === "match") {
				addSummaryMatch(parsed.displayPath, absolute);
			}
			continue;
		}
		const fileLines = await getFileLines(absolute);
		if (fileLines === undefined) continue;
		// Bare-CR remapping: rg treats the entire bare-CR file as line 1, and the
		// builtin grep tool may strip \r before this code sees the output. So
		// parsed.text is just the first CR-separated fragment and parsed.lineNumber
		// is always 1 — both are wrong for match lines. Only remap when
		// parsed.kind === "match"; context lines are irrelevant here (rg won’t
		// produce them for bare-CR files in any meaningful way).
		if (parsed.kind === "match" && bareCRFiles.has(absolute)) {
			const flags = p.ignoreCase ? "i" : "";
			let patternRe: RegExp | null = null;
			try {
				patternRe = p.literal
					? new RegExp(escapeForRegex(p.pattern), flags)
					: new RegExp(p.pattern, flags);
			} catch {
				// Malformed regex — fall through to normal anchor path
			}
			if (patternRe !== null) {
				let emitted = false;
				for (let i = 0; i < fileLines.length; i++) {
					if (!patternRe.test(fileLines[i])) continue;
					addEntry(parsed.displayPath, absolute, "match", buildReadSeekLine(i + 1, fileLines[i]));
					emitted = true;
				}
				if (emitted) continue;
				// No lines matched — fall through to normal path
			}
		}
		// Normal (non-bare-CR) path
		const sourceLine = fileLines[parsed.lineNumber - 1] ?? parsed.text;
		const built = buildReadSeekLine(parsed.lineNumber, sourceLine);
		addEntry(parsed.displayPath, absolute, parsed.kind, {
			...built,
			display: escapeControlCharsForDisplay(parsed.text),
		});
	}

	if (p.summary && parsedCount === 0 && candidateUnparsedCount > 0) {
		const passthroughDetails =
			typeof result.details === "object" && result.details !== null
				? (result.details as Record<string, unknown>)
				: {};
		return {
			...result,
			details: {
				...passthroughDetails,
				readSeekValue: {
					tool: "grep",
					summary: true,
					totalMatches: 0,
					records: [],
				},
			},
		};
	}

	if (parsedCount === 0 && candidateUnparsedCount > 0) {
		const warning =
			"[hashline grep passthrough] Unparsed grep format; returned original output.";
		const passthroughDetails =
			typeof result.details === "object" && result.details !== null
				? (result.details as Record<string, unknown>)
				: {};
		return {
			...result,
			content: result.content.map((item) =>
				item === textBlock ? ({ ...item, text: `${textBlock.text}\n\n${warning}` } as typeof item) : item,
			),
			details: {
				...passthroughDetails,
				hashlinePassthrough: true,
				hashlineWarning: warning,
				readSeekValue: {
					tool: "grep",
					summary: !!p.summary,
					totalMatches: 0,
					records: [],
				},
			},
		};
	}

	const summary = !!p.summary;
	const effectiveLimit = typeof p.limit === "number" ? p.limit : 100;
	const groups = [...groupsByPath.values()];
	for (const group of groups) {
		group.entries = dedupeContextEntries(group.entries);
	}
	let renderedGroups: GrepOutputGroup[] =
		totalMatches > GREP_TRUNCATION_THRESHOLD ? groups.map(truncateGroupEntries) : groups;
	let scopeWarnings: GrepScopeWarning[] = [];

	if (p.scope === "symbol" && !summary) {
		const fileLinesByPath = new Map<string, string[]>();
		const fileMapsByPath = new Map<string, Awaited<ReturnType<typeof getOrGenerateMap>>>();

		for (const group of renderedGroups) {
			const lines = await getFileLines(group.absolutePath);
			if (lines) fileLinesByPath.set(group.absolutePath, lines);
			fileMapsByPath.set(group.absolutePath, await getOrGenerateMap(group.absolutePath));
		}

		const scoped = scopeGrepGroupsToSymbols({
			groups: renderedGroups,
			fileLinesByPath,
			fileMapsByPath,
			contextLines: typeof p.context === "number" ? p.context : 0,
			scopeContext: typeof p.scopeContext === "number" ? p.scopeContext : undefined,
		});

		renderedGroups = scoped.groups;
		scopeWarnings = scoped.warnings;
	}
	const readSeekRecords = recordsFromGroups(renderedGroups);
	const builtOutput = buildGrepOutput({
		summary: !!summary,
		totalMatches,
		groups: renderedGroups,
		limit: effectiveLimit,
		records: readSeekRecords,
		scopeMode: p.scope === "symbol" && !summary ? "symbol" : undefined,
		scopeWarnings,
		passthroughLines,
	});

	if (!summary && readSeekRecords.length > 0) {
		const anchoredPaths = new Set(readSeekRecords.map((record) => record.path));
		for (const absolutePath of anchoredPaths) {
			onFileAnchored?.(absolutePath);
		}
	}

	const existingDetails =
		typeof result.details === "object" && result.details !== null
			? (result.details as Record<string, unknown>)
			: {};
	const { linesTruncated: _ignoredLinesTruncated, truncation: _ignoredTruncation, ...compactDetails } = existingDetails;
	return {
		...result,
		content: result.content.map((item) =>
			item === textBlock ? ({ ...item, text: builtOutput.text } as typeof item) : item,
		),
		details: {
			...compactDetails,
			readSeekValue: builtOutput.readSeekValue,
		},
	};
}

export function registerGrepTool(pi: ExtensionAPI, options: GrepToolOptions = {}) {
	const name = options.name ?? "readSeek_grep";
	const promptMetadata = defineToolPromptMetadata({
		promptUrl: new URL("../prompts/grep.md", import.meta.url),
		promptSnippet: "Search plain text or regex with edit-ready anchors",
		registeredName: name,
	});
	const tool = registerReadSeekTool(pi, {
		name,
		label: "grep",
		description: promptMetadata.description,
		parameters: grepSchema,
		promptSnippet: promptMetadata.promptSnippet,
		promptGuidelines: promptMetadata.promptGuidelines,
		async execute(toolCallId, params, signal, onUpdate, ctx) {
			return executeGrep({
				toolCallId,
				params,
				signal,
				onUpdate,
				cwd: ctx.cwd,
				onFileAnchored: options.onFileAnchored,
			});
		},
		renderCall(args: any, theme: any, ...rest: any[]) {
			const context = rest[0] ?? {};
			const cwd = context.cwd ?? process.cwd();
			const { pattern, suffix } = formatGrepCallText(args);
			const rawPath = typeof args?.path === "string" && args.path !== "." ? args.path : undefined;
			const glob = typeof args?.glob === "string" ? args.glob : undefined;
			let text = `${renderToolLabel(theme, "grep")} ${theme.fg("accent", `/${pattern}/`)}`;
			if (suffix) {
				if (rawPath) {
					text += theme.fg("dim", " in ");
					text += linkToolPath(theme.fg("dim", rawPath), rawPath, cwd);
					if (glob) text += theme.fg("dim", ` ${glob}`);
				} else {
					text += theme.fg("dim", ` in ${suffix}`);
				}
			}
			return new Text(clampLineToWidth(text, context.width), 0, 0);
		},
		renderResult(result: any, options: ToolRenderResultOptions, theme: any, ...rest: any[]) {
			const { isPartial, isError, expanded, cwd, width } = resolveRenderResultContext(options, rest);

			if (isPartial) return renderPendingResult("pending grep", width, theme);

			const content = result.content?.[0];
			const textContent = content?.type === "text" ? content.text : "";

			if (isError || result.isError) return renderErrorResult(textContent, { expanded, width, theme });

			const readSeekValue = (result.details as any)?.readSeekValue as {
				tool: "grep";
				summary: boolean;
				totalMatches: number;
				records: Array<{ path: string; anchor: string; kind: string }>;
			} | undefined;

			const hasBinaryWarning = textContent.includes("appears to be a binary file");

			const fileSet = new Set<string>();
			for (const r of readSeekValue?.records ?? []) {
				if (r.path) fileSet.add(r.path);
			}

			const info = formatGrepResultText({
				totalMatches: readSeekValue?.totalMatches ?? 0,
				summary: readSeekValue?.summary ?? false,
				records: readSeekValue?.records ?? [],
				fileCount: fileSet.size,
				hasBinaryWarning,
			});

			if (info.noMatches && !hasBinaryWarning) return new Text(summaryLine("no matches", { theme, style: "dim" }), 0, 0);
			const matchCount = readSeekValue?.totalMatches ?? 0;
			const matchWord = matchCount === 1 ? "match" : "matches";
			let text = summaryLine(`${matchCount} ${matchWord} returned`, {
				hidden: !!textContent && !expanded,
				theme,
				style: "success",
			});
			for (const badge of info.badges) text += theme.fg(badge.startsWith("⚠") ? "warning" : "dim", `  ${badge}`);
			if (expanded && textContent && readSeekValue?.records) {
				const anchors = new Set(readSeekValue.records.map((record) => record.anchor));
				text += "\n" + renderGrepSourceForDisplay(textContent, anchors, (displayPath) => linkToolPath(theme.fg("dim", displayPath), displayPath, cwd));
			}
			return new Text(clampLinesToWidth(text.split("\n"), width).join("\n"), 0, 0);
		},
	});
	return tool;
}
