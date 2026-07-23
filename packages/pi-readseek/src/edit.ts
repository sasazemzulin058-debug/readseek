import { readFile as fsReadFile, writeFile as fsWriteFile } from "node:fs/promises";

import { createPatch } from "diff";
import { withFileMutationQueue, type ExtensionAPI, type EditToolDetails, type ToolRenderResultOptions } from "@earendil-works/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import type { Static } from "@sinclair/typebox";
import { Text } from "@earendil-works/pi-tui";

import { defineToolPromptMetadata } from "./tool-prompt-metadata.js";
import { detectLineEnding, generateCompactOrFullDiff, normalizeToLF, replaceText, restoreLineEndings, stripBom } from "./edit-diff.js";
import { HashlineMismatchError, applyHashlineEdits, computeLineHash, ensureHashInit, parseLineRef, type HashlineEditItem, escapeControlCharsForDisplay } from "./hashline.js";
import { resolveToCwd } from "./path-utils.js";
import { looksLikeBinary } from "./binary-detect.js";
import { throwIfAborted } from "./runtime.js";
import { formatFsError } from "./fs-error.js";
import { buildEditOutput } from "./edit-output.js";
import { classifyEdit } from "./edit-classify.js";
import type { SemanticSummary } from "./readseek-value.js";
import { buildReadSeekError } from "./readseek-value.js";
import { classifyReadSeekFailure } from "./readseek-client.js";
import { countEditTypes, formatEditCallText, formatEditResultText } from "./edit-render-helpers.js";
import { validateSyntaxRegression } from "./edit-syntax-validate.js";
import { resolveSyntaxValidateMode, type SyntaxValidateOptions } from "./syntax-validate-mode.js";
import { replaceSymbol } from "./replace-symbol.js";
import { buildEditPreviewKey, buildPendingEditPreviewData, resolvePendingDiffPreview, type PendingDiffPreviewResult } from "./pending-diff-preview.js";
import { buildDiffData, type DiffBlockRange } from "./diff-data.js";
import { clampLineToWidth, clampLinesToWidth, linkToolPath, renderPendingResult, resolveRenderResultContext, summaryLine } from "./tui-render-utils.js";
import { resolveReadSeekToolDisplayMode } from "./readseek-settings.js";
import { upsertDiffComponent, upsertTextComponent } from "./tui-diff-component.js";
import type { FileMutatedCallback, FreshAnchorsPredicate } from "./tool-types.js";
import { filePathParam, registerReadSeekTool } from "./register-tool.js";

const EDIT_PENDING_PREVIEW_STATE_KEY = "hashline-edit-pending-preview";

function pendingPreviewLines(summary: string, preview: PendingDiffPreviewResult | undefined, expanded: boolean): { lines: string[]; diffData?: ReturnType<typeof buildDiffData>; headerLabel?: string } {
	if (!preview || preview.type !== "ok") return { lines: summary.split("\n") };
	const diffData = buildDiffData({ path: preview.data.filePath, oldContent: preview.data.previousContent, newContent: preview.data.nextContent, diff: preview.data.diff });
	const headerLine = summaryLine(preview.data.headerLabel, { hidden: !expanded });
	return { lines: [summary, headerLine], diffData: expanded ? diffData : undefined, headerLabel: preview.data.headerLabel };
}


const hashlineEditItemSchema = Type.Union([
	Type.Object({ set_line: Type.Object({ anchor: Type.String(), new_text: Type.String() }) }, { additionalProperties: false }),
	Type.Object(
		{ replace_lines: Type.Object({ start_anchor: Type.String(), end_anchor: Type.String(), new_text: Type.String() }) },
		{ additionalProperties: false },
	),
	Type.Object({ insert_after: Type.Object({ anchor: Type.String(), new_text: Type.String() }) }, { additionalProperties: false }),
	Type.Object(
		{ replace: Type.Object({ old_text: Type.String(), new_text: Type.String(), all: Type.Optional(Type.Boolean()), fuzzy: Type.Optional(Type.Boolean()) }) },
		{ additionalProperties: false },
	),
	Type.Object(
		{ replace_symbol: Type.Object({ symbol: Type.String(), new_body: Type.String() }) },
		{ additionalProperties: false },
	),
]);

const hashlineEditSchema = Type.Object(
	{
		path: filePathParam(),
		edits: Type.Optional(Type.Array(hashlineEditItemSchema, {
			description: "Edits: set_line, replace_lines, insert_after, replace_symbol, or replace",
		})),
		postEditVerify: Type.Optional(Type.Boolean({
			description: "Verify persisted content after write",
		})),
	},
	{ additionalProperties: true },
);

type HashlineParams = Static<typeof hashlineEditSchema>;


function buildEditError(
	path: string,
	code: string,
	message: string,
	hint?: string,
	errorDetails?: Record<string, unknown>,
): {
	content: [{ type: "text"; text: string }];
	isError: true;
	details: EditToolDetails & { readSeekValue: any };
} {
	return {
		content: [{ type: "text", text: message }],
		isError: true,
		details: {
			diff: "",
			patch: "",
			firstChangedLine: undefined,
			readSeekValue: {
				tool: "edit",
				ok: false,
				path,
				error: buildReadSeekError(code, message, hint, errorDetails),
			},
		} as EditToolDetails & { readSeekValue: any },
	};
}

function mapEditFileError(err: any, filePath: string, _displayPath: string, _phase: "read" | "write"): ReturnType<typeof buildEditError> {
	const { code, message } = formatFsError(err, "edit-error");
	return buildEditError(filePath, code, message);
}

export interface EditToolOptions {
	wasReadInSession?: FreshAnchorsPredicate;
	onFileMutated?: FileMutatedCallback;
	syntaxValidate?: SyntaxValidateOptions["syntaxValidate"];
	name?: string;
	toolAliases?: Readonly<Record<string, string>>;
}

export interface ExecuteEditOptions {
	params: unknown;
	signal: AbortSignal | undefined;
	cwd: string;
	wasReadInSession?: FreshAnchorsPredicate;
	onFileMutated?: FileMutatedCallback;
	syntaxValidate?: SyntaxValidateOptions["syntaxValidate"];
}

export async function executeEdit(opts: ExecuteEditOptions): Promise<any> {
	const { params, signal, cwd, wasReadInSession, onFileMutated, syntaxValidate } = opts;
	await ensureHashInit();
	const parsed = params as HashlineParams;
	const input = params as Record<string, unknown>;
	const rawPath = parsed.path;
	const path = rawPath.replace(/^@/, "");
	const absolutePath = resolveToCwd(path, cwd);
	throwIfAborted(signal);
	try {
		return await withFileMutationQueue(absolutePath, async () => {
		throwIfAborted(signal);
	if (wasReadInSession && !wasReadInSession(absolutePath)) {
		const message = [
			`You must get fresh anchors for ${absolutePath} before editing it.`,
			`Call readSeek_read(${JSON.stringify(rawPath)}) first, or use readSeek_grep, readSeek_search, or readSeek_write to produce fresh anchors for this file.`,
			"readSeek_edit requires fresh LINE:HASH anchors from readSeek_read, readSeek_grep, readSeek_search, or readSeek_write so the hashes match the current file contents.",
		].join(" ");
		return buildEditError(
			absolutePath,
			"file-not-read",
			message,
			`Call readSeek_read(${JSON.stringify(rawPath)}) first, or use readSeek_grep, readSeek_search, or readSeek_write to produce fresh anchors for this file.`,
		);
	}
	const hasTopLevelReplaceInput =
		typeof input.oldText === "string" ||
		typeof input.newText === "string" ||
		typeof input.old_text === "string" ||
		typeof input.new_text === "string";
	if (hasTopLevelReplaceInput) {
		return buildEditError(
			absolutePath,
			"invalid-edit-variant",
			"Top-level oldText/newText and old_text/new_text are no longer supported. Use edits[0].replace instead.",
			"Use edits: [{ replace: { old_text, new_text } }].",
		);
	}

	const edits = parsed.edits ?? [];

	if (!edits.length) {
		return buildEditError(absolutePath, "invalid-edit-variant", "No edits provided.");
	}

	// Validate edit variant keys
	for (let i = 0; i < edits.length; i++) {
		throwIfAborted(signal);
		const e = edits[i] as Record<string, unknown>;
		if (("old_text" in e || "new_text" in e) && !("replace" in e)) {
			const message = `edits[${i}] has top-level 'old_text'/'new_text'. Use {replace: {old_text, new_text}} or {set_line}, {replace_lines}, {insert_after}.`;
			return buildEditError(absolutePath, "invalid-edit-variant", message);
		}
		if ("diff" in e) {
			const message = `edits[${i}] contains 'diff' from patch mode. Hashline edit expects one of: {set_line}, {replace_lines}, {insert_after}, {replace}.`;
			return buildEditError(absolutePath, "invalid-edit-variant", message);
		}
		const variantCount =
			Number("set_line" in e) +
			Number("replace_lines" in e) +
			Number("insert_after" in e) +
			Number("replace" in e) +
			Number("replace_symbol" in e);
		if (variantCount !== 1) {
			const message = `edits[${i}] must contain exactly one of: 'set_line', 'replace_lines', 'insert_after', 'replace', 'replace_symbol'. Got: [${Object.keys(e).join(", ")}].`;
			return buildEditError(absolutePath, "invalid-edit-variant", message);
		}
	}

	const anchorEdits = edits.filter(
		(e): e is HashlineEditItem => "set_line" in e || "replace_lines" in e || "insert_after" in e,
	);
	const replaceEdits = edits.filter(
		(e): e is { replace: { old_text: string; new_text: string; all?: boolean; fuzzy?: boolean } } => "replace" in e,
	);
	const replaceSymbolEdits = edits.filter(
		(e): e is { replace_symbol: { symbol: string; new_body: string } } => "replace_symbol" in e,
	);
	for (const rs of replaceSymbolEdits) {
		if (!rs.replace_symbol.new_body.trim()) {
			return buildEditError(absolutePath, "invalid-edit-variant", "replace_symbol.new_body must not be empty or whitespace-only.");
		}
	}

	let rawBuffer: Buffer;
	try {
		rawBuffer = await fsReadFile(absolutePath);
	} catch (err: any) {
		return mapEditFileError(err, absolutePath, path, "read");
	}
	if (looksLikeBinary(rawBuffer)) {
		const message = `Cannot edit binary file: ${path}`;
		return buildEditError(absolutePath, "binary-file", message);
	}
	throwIfAborted(signal);
	const raw = rawBuffer.toString("utf-8");
	const { bom, text: content } = stripBom(raw);
	const originalEnding = detectLineEnding(content);
	const originalNormalized = normalizeToLF(content);
	const origLines = originalNormalized.split("\n");
	const anchorSnapshots = new Map<string, string>();
	for (const edit of anchorEdits) {
		const refs: string[] = [];
		if ("set_line" in edit) refs.push(edit.set_line.anchor);
		else if ("replace_lines" in edit) {
			refs.push(edit.replace_lines.start_anchor, edit.replace_lines.end_anchor);
		} else if ("insert_after" in edit) refs.push(edit.insert_after.anchor);
		for (const ref of refs) {
			try {
				const parsed = parseLineRef(ref);
				if (parsed.line >= 1 && parsed.line <= origLines.length) {
					const lineContent = origLines[parsed.line - 1];
					const hash = computeLineHash(lineContent);
					anchorSnapshots.set(ref, `${parsed.line}:${hash}|${escapeControlCharsForDisplay(lineContent)}`);
				}
			} catch {
				/* skip malformed refs */
			}
		}
	}
	let preAnchorContent = originalNormalized;
	// reject anchored edits that target a line inside any replace_symbol
	// pre-replace range. Resolve each target against the ORIGINAL content so the
	// user-provided anchor line numbers (which reference the file as read) are
	// compared against the pre-replace coordinates.
	//
	// surface replace_symbol symbol-resolution errors (not-found, ambiguous)
	// before the overlap check and before any write.
	// Error-precedence order: replace_symbol resolution > anchor-overlap > anchored-edit.
	//
	// store successful probe results and reuse them in the apply loop so
	// readSeekMapContent is invoked at most once per replace_symbol edit.
	const replaceSymbolRanges: { start: number; end: number }[] = [];
	const rsProbeResults: { type: "ok"; content: string; replacement: string; warnings: string[]; range: { start: number; end: number } }[] = [];
	try {
		for (const rs of replaceSymbolEdits) {
			throwIfAborted(signal);
			const probe = await replaceSymbol({
				filePath: absolutePath,
				content: originalNormalized,
				symbol: rs.replace_symbol.symbol,
				newBody: rs.replace_symbol.new_body,
			});
			if (probe.type !== "ok") {
				// symbol-resolution errors surface before the overlap check.
				return buildEditError(absolutePath, "invalid-edit-variant", probe.message);
			}
			rsProbeResults.push(probe);
			replaceSymbolRanges.push(probe.range);
		}
	} catch (err) {
		throwIfAborted(signal);
		const failure = classifyReadSeekFailure(err);
		return buildEditError(absolutePath, failure.code, failure.message, failure.hint);
	}

	const sortedReplaceSymbolRanges = [...replaceSymbolRanges].sort((a, b) => a.start - b.start || a.end - b.end);
	for (let i = 1; i < sortedReplaceSymbolRanges.length; i++) {
		const prev = sortedReplaceSymbolRanges[i - 1];
		const current = sortedReplaceSymbolRanges[i];
		if (current.start <= prev.end) {
			const message = `replace_symbol ranges overlap or duplicate (lines ${prev.start}-${prev.end} and ${current.start}-${current.end}).`;
			return buildEditError(absolutePath, "invalid-edit-variant", message);
		}
	}
	if (replaceSymbolRanges.length > 0) {
		for (const edit of anchorEdits) {
			if ("replace_lines" in edit) {
				let startLine: number | undefined;
				let endLine: number | undefined;
				try {
					startLine = parseLineRef(edit.replace_lines.start_anchor).line;
					endLine = parseLineRef(edit.replace_lines.end_anchor).line;
				} catch {
					// Let the normal anchored edit validation report malformed anchors later.
				}
				if (startLine !== undefined && endLine !== undefined) {
					const lo = Math.min(startLine, endLine);
					const hi = Math.max(startLine, endLine);
					for (const range of replaceSymbolRanges) {
						if (lo <= range.end && hi >= range.start) {
							const message = `replace_lines range ${lo}-${hi} overlaps a replace_symbol range (lines ${range.start}-${range.end}).`;
				return buildEditError(absolutePath, "invalid-edit-variant", message);
						}
					}
				}
			}
			const refs: string[] = [];
			if ("set_line" in edit) refs.push(edit.set_line.anchor);
			else if ("replace_lines" in edit) {
				refs.push(edit.replace_lines.start_anchor, edit.replace_lines.end_anchor);
			} else if ("insert_after" in edit) refs.push(edit.insert_after.anchor);
			for (const ref of refs) {
				let parsedLine: number | undefined;
				try {
					parsedLine = parseLineRef(ref).line;
				} catch {
					continue;
				}
				for (const range of replaceSymbolRanges) {
					if (parsedLine >= range.start && parsedLine <= range.end) {
						const message = `Anchor at line ${parsedLine} falls inside a replace_symbol range (lines ${range.start}-${range.end}).`;
				return buildEditError(absolutePath, "invalid-edit-variant", message);
					}
				}
			}
		}
	}

	const anchorIntervals: { lo: number; hi: number }[] = [];
	for (const edit of anchorEdits) {
		try {
			if ("set_line" in edit) {
				const line = parseLineRef(edit.set_line.anchor).line;
				anchorIntervals.push({ lo: line, hi: line });
			} else if ("replace_lines" in edit) {
				const start = parseLineRef(edit.replace_lines.start_anchor).line;
				const end = parseLineRef(edit.replace_lines.end_anchor).line;
				anchorIntervals.push({ lo: Math.min(start, end), hi: Math.max(start, end) });
			}
		} catch {
		}
	}
	const seenAnchorIntervals = new Set<string>();
	const uniqueAnchorIntervals: { lo: number; hi: number }[] = [];
	for (const interval of anchorIntervals) {
		const key = `${interval.lo}:${interval.hi}`;
		if (seenAnchorIntervals.has(key)) continue;
		seenAnchorIntervals.add(key);
		uniqueAnchorIntervals.push(interval);
	}
	uniqueAnchorIntervals.sort((a, b) => a.lo - b.lo || a.hi - b.hi);
	for (let i = 1; i < uniqueAnchorIntervals.length; i++) {
		const prev = uniqueAnchorIntervals[i - 1];
		const current = uniqueAnchorIntervals[i];
		if (current.lo <= prev.hi) {
			const message = `Anchored edits overlap (lines ${prev.lo}-${prev.hi} and ${current.lo}-${current.hi}). Split into separate, non-overlapping edits or re-read for fresh anchors.`;
			return buildEditError(absolutePath, "invalid-edit-variant", message);
		}
	}
	// Apply pass: reuse all probe results. The probe pass resolved every
	// replace_symbol against originalNormalized; apply those replacements in
	// reverse source order so original line ranges stay valid and no second
	// replaceSymbol/readSeekMapContent call is needed.
	const replaceSymbolWarnings: string[] = [];
	if (rsProbeResults.length > 0) {
		const lines = originalNormalized.split("\n");
		for (const probe of rsProbeResults) {
			replaceSymbolWarnings.push(...probe.warnings);
		}
		for (const probe of [...rsProbeResults].sort((a, b) => b.range.start - a.range.start)) {
			lines.splice(
				probe.range.start - 1,
				probe.range.end - probe.range.start + 1,
				...probe.replacement.split("\n"),
			);
		}
		preAnchorContent = lines.join("\n");
	}
	let result = preAnchorContent;

	let anchorResult;
	try {
		anchorResult = applyHashlineEdits(result, anchorEdits, signal);
	} catch (err) {
		if (err instanceof HashlineMismatchError) {
			return buildEditError(absolutePath, "hash-mismatch", err.message, undefined, {
				updatedAnchors: err.updatedAnchors,
			});
		}
		throw err;
	}
	result = anchorResult.content;

	const replaceWarnings: string[] = [];
	for (const r of replaceEdits) {
		throwIfAborted(signal);
		if (!r.replace.old_text.length) {
			const message = "replace.old_text must not be empty.";
			return buildEditError(absolutePath, "invalid-edit-variant", message);
		}
		const rep = replaceText(result, r.replace.old_text, r.replace.new_text, {
			all: r.replace.all ?? false,
			fuzzy: r.replace.fuzzy ?? false,
		});
		if (!rep.count) {
			const message = `Could not find exact text to replace in ${path}.`;
			const hint =
				"Re-read the file and prefer set_line/replace_lines/insert_after for hash-verified edits. " +
				"The replace variant is exact-only by default because fuzzy fallback is unverified.";
			return buildEditError(absolutePath, "text-not-found", message, hint);
		}
		if (rep.usedFuzzyMatch) {
			replaceWarnings.push(
				"replace used fuzzy matching because exact old_text was not found; re-read the file and prefer set_line/replace_lines/insert_after for hash-verified edits.",
			);
		}
		result = rep.content;
	}

	if (originalNormalized === result) {
		let diagnostic = `No changes made to ${path}. The edits produced identical content.`;
		if (anchorResult.noopEdits?.length) {
			diagnostic +=
				"\n" +
				anchorResult.noopEdits
					.map(
						(e) =>
							`Edit ${e.editIndex}: replacement for ${e.loc} is identical to current content:\n  ${e.loc}| ${escapeControlCharsForDisplay(e.currentContent)}`,
					)
					.join("\n");
			diagnostic += "\nRe-read the file to see the current state.";
		} else {
			const targetLines = [...new Set(anchorSnapshots.values())];
			if (targetLines.length > 0) {
				const preview = targetLines.slice(0, 5).join("\n");
				diagnostic += `\nThe file currently contains:\n${preview}\nYour edits were normalized back to the original content. Ensure your replacement changes actual code, not just formatting.`;
			}
		}
		return buildEditError(absolutePath, "no-op", diagnostic);
	}

	throwIfAborted(signal);

	// Syntax-regression validator (warn/block/off)
	const syntaxMode = resolveSyntaxValidateMode({ syntaxValidate: syntaxValidate });
	let syntaxWarning: string | undefined;
	if (syntaxMode !== "off") {
		const regression = await validateSyntaxRegression({
			filePath: absolutePath,
			before: originalNormalized,
			after: result,
		}, { signal });
		if (regression) {
			const lines = regression.errorLines.join(", ");
			const message = `syntax-regression: lines ${lines}`;
			// block mode aborts with syntax-regression code; file is left untouched.
			if (syntaxMode === "block") {
				return buildEditError(absolutePath, "syntax-regression", message);
			}
			syntaxWarning = message;
		}
	}
	const writeContent = bom + restoreLineEndings(result, originalEnding);
	try {
		await fsWriteFile(absolutePath, writeContent, "utf-8");
	} catch (err: any) {
		return mapEditFileError(err, absolutePath, path, "write");
	}
	onFileMutated?.(absolutePath);

	if (input.postEditVerify === true) {
		let verifiedContent: string;
		try {
			const verified = await fsReadFile(absolutePath, "utf-8");
			verifiedContent = verified;
		} catch (err: any) {
			const message = `Edit write completed but post-edit verification failed: could not read ${path} after writing.`;
			return buildEditError(absolutePath, "post-edit-verification-read-failed", message, undefined, {
					fsCode: err?.code,
					fsMessage: err?.message,
				},
			);
		}
		if (verifiedContent !== writeContent) {
			const message = `Edit write completed but post-edit verification did not confirm the intended content for ${path}. Re-read the file before making follow-up edits.`;
			return buildEditError(absolutePath, "post-edit-verification-mismatch", message, undefined, {
					expectedLength: writeContent.length,
					actualLength: verifiedContent.length,
				},
			);
		}
	}

	const diffResult = generateCompactOrFullDiff(originalNormalized, result);
	const patch = createPatch(path, originalNormalized, result);
	const blockRanges: DiffBlockRange[] = rsProbeResults.map((probe) => ({
		kind: "remove" as const,
		startLine: probe.range.start,
		endLine: probe.range.end,
	}));
	const diffData = buildDiffData({
		path: absolutePath,
		oldContent: originalNormalized,
		newContent: result,
		diff: diffResult.diff,
		...(blockRanges.length ? { blockRanges } : {}),
	});
	const warnings: string[] = [];
	if (anchorResult.warnings?.length) warnings.push(...anchorResult.warnings);
	if (replaceWarnings.length) warnings.push(...replaceWarnings);
	if (replaceSymbolWarnings.length) warnings.push(...replaceSymbolWarnings);
	if (syntaxWarning) warnings.push(syntaxWarning);
	// Semantic classification
	const internalClassification = classifyEdit(originalNormalized, result);
	const semanticSummary: SemanticSummary = {
		classification: internalClassification.classification,
	};
	const builtOutput = buildEditOutput({
		path: absolutePath,
		displayPath: path,
		diff: diffResult.diff,
		patch,
		diffData,
		firstChangedLine: anchorResult.firstChangedLine ?? diffResult.firstChangedLine,
		warnings,
		noopEdits: anchorResult.noopEdits ?? [],
		edits,
		semanticSummary,
	});

	return {
		content: [{ type: "text", text: builtOutput.text }],
		details: {
			diff: diffResult.diff,
			patch: builtOutput.patch,
			diffData,
			firstChangedLine: anchorResult.firstChangedLine ?? diffResult.firstChangedLine,
			readSeekValue: builtOutput.readSeekValue,
		} as EditToolDetails & {
			diffData: typeof diffData;
			readSeekValue: {
				tool: string;
				ok: boolean;
				path: string;
				summary: string;
				diff: string;
				diffData: typeof diffData;
				firstChangedLine: number | undefined;
				warnings: string[];
				noopEdits: unknown[];
			};
		},
	};
		});
	} catch (err: any) {
		if (typeof err?.code === "string") {
			return mapEditFileError(err, absolutePath, path, "read");
		}
		throw err;
	}
}


export function registerEditTool(pi: ExtensionAPI, options: EditToolOptions = {}) {
	const name = options.name ?? "readSeek_edit";
	const promptMetadata = defineToolPromptMetadata({
		promptUrl: new URL("../prompts/edit.md", import.meta.url),
		promptSnippet: "Safely edit with fresh LINE:HASH anchors",
		registeredName: name,
		toolAliases: options.toolAliases,
	});
	const tool = registerReadSeekTool(pi, {
		name,
		label: "Edit",
		description: promptMetadata.description,
		promptSnippet: promptMetadata.promptSnippet,
		promptGuidelines: promptMetadata.promptGuidelines,
		parameters: hashlineEditSchema,
		renderShell: "default" as const,
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			return executeEdit({
				params,
				signal,
				cwd: ctx.cwd,
				wasReadInSession: options.wasReadInSession,
				onFileMutated: options.onFileMutated,
				syntaxValidate: options.syntaxValidate,
			});
		},
		renderCall(args: any, theme: any, ...rest: any[]) {
			const context: { argsComplete?: boolean; executionStarted?: boolean; lastComponent?: any; cwd?: string; state?: Record<string, any>; invalidate?: () => void; width?: number; expanded?: boolean } = rest[0] ?? {};
			const cwd = context.cwd ?? process.cwd();
			const argsComplete = context.argsComplete ?? false;
			const { path: filePath, suffix } = formatEditCallText(args, argsComplete);

			let text = theme.fg("toolTitle", theme.bold("edit"));
			if (filePath) text += ` ${linkToolPath(theme.fg("accent", filePath), filePath, cwd)}`;
			else text += ` ${theme.fg("toolOutput", "...")}`;
			const counts = Array.isArray(args?.edits) ? countEditTypes(args.edits) : undefined;
			if (counts && counts.total > 0) {
				text += ` ${theme.fg("dim", `(${counts.total} ${counts.total === 1 ? "edit" : "edits"})`)}`;
			} else if (suffix) {
				text += ` ${theme.fg("dim", suffix)}`;
			}
			text = clampLineToWidth(text, context.width);
			// Once execution has started, the pending preview's only job is done:
			// renderResult will carry the story ("↳ edited +N -M" with the same
			// expandable diff). Keeping the "↳ pending edit" sub-line and its
			// preview alongside the final result is just duplicate noise.
			if (context.executionStarted) {
				return upsertTextComponent(context.lastComponent, text);
			}
			const previewKey = buildEditPreviewKey(args ?? {});
			const preview = resolvePendingDiffPreview(
				context,
				EDIT_PENDING_PREVIEW_STATE_KEY,
				previewKey,
				() => buildPendingEditPreviewData(args ?? {}, context.cwd ?? process.cwd()),
			);
			const expanded = !!context.expanded;
			const preview2 = pendingPreviewLines(text, preview, expanded);
			if (preview2.diffData) {
				return upsertDiffComponent(context.lastComponent, { prefixLines: preview2.lines, diffData: preview2.diffData, theme, expanded: true });
			}
			return upsertTextComponent(context.lastComponent, clampLinesToWidth(preview2.lines, context.width).join("\n"));
		},
			renderResult(result: any, options: ToolRenderResultOptions, theme: any, ...rest: any[]) {
			const { isPartial, isError, expanded, width, context } = resolveRenderResultContext(
				options,
				rest,
				resolveReadSeekToolDisplayMode("edit") === "expanded",
			);

			if (isPartial) {
				return renderPendingResult("pending edit", width, theme);
			}

			// Extract data from result
			const textContent = result.content
				?.filter((c: any) => c.type === "text")
				.map((c: any) => c.text || "")
				.join("\n") ?? "";
			const details = result.details ?? {};
			const diff: string = details.diff ?? "";
			const readSeekValue = details.readSeekValue as {
				warnings?: string[];
				noopEdits?: unknown[];
			} | undefined;
			const warnings = readSeekValue?.warnings ?? [];
			const noopEdits = readSeekValue?.noopEdits ?? [];
			const semanticClassification = (readSeekValue as any)?.semanticSummary?.classification as string | undefined;

			const info = formatEditResultText({
				isError: isError || !!result.isError,
				diff,
				warnings,
				noopEdits,
				errorText: textContent,
				semanticClassification: semanticClassification as any,
			});

			const diffData = (details as any).diffData;
			const stats = diffData?.stats ?? { added: 0, removed: 0 };
			let text = "";

			if (info.noOp) {
				text = summaryLine("no-op", { theme, style: "dim" });
				if (expanded && info.errorText) text += `\n${theme.fg("error", info.errorText)}`;
			} else if (info.errorText) {
				const firstLine = info.errorText.split("\n")[0] || "Error";
				text = summaryLine(expanded ? info.errorText : firstLine, { theme, style: "error" });
			} else {
				const badges: string[] = [`edited +${stats.added} -${stats.removed}`];
				if (info.semanticBadge) badges.push(info.semanticBadge.replace(/^✓\s*/, ""));
				if (info.warningsBadge) badges.push(info.warningsBadge);
				text = summaryLine(badges.join(" • "), { hidden: !!diffData && !expanded, theme, style: "success" });
				if (expanded && diffData) {
					return upsertDiffComponent(context.lastComponent, { prefixLines: text.split("\n"), diffData, theme, expanded: true });
				}
			}
			return new Text(clampLinesToWidth(text.split("\n"), width).join("\n"), 0, 0);
		},
	});
	return tool;
}
