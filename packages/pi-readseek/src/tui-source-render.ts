import { getLanguageFromPath, highlightCode } from "@earendil-works/pi-coding-agent";
import { wrapTextWithAnsi } from "@earendil-works/pi-tui";

import { normalizeWidth } from "./tui-render-utils.js";

const READ_ANCHOR_RE = /^(\d+:[0-9a-zA-Z]{1,16})\|(.*)$/;
const GREP_ANCHOR_RE = /^(.*?):(>>|  )(\d+):([0-9a-zA-Z]{1,16})\|(.*)$/;

const SOURCE_THEME_STYLES = [
	"mdCodeBlock",
	"muted",
	"syntaxComment",
	"syntaxKeyword",
	"syntaxFunction",
	"syntaxVariable",
	"syntaxString",
	"syntaxNumber",
	"syntaxType",
	"syntaxOperator",
	"syntaxPunctuation",
	"toolDiffAdded",
	"toolDiffRemoved",
] as const;

interface SourceRenderTheme {
	fg(style: string, text: string): string;
}

function wrapDisplayLines(lines: string[], width: number | undefined): string[] {
	if (width === undefined || width === null) return lines;
	const normalized = normalizeWidth(width);
	return lines.flatMap((line) => {
		const wrapped = wrapTextWithAnsi(line, normalized);
		return wrapped.length > 0 ? wrapped : [""];
	});
}

function renderSourceBlock(lines: string[], path: string): string[] {
	return highlightCode(lines.join("\n"), getLanguageFromPath(path));
}

export function renderReadSourceForDisplay(
	text: string,
	path: string,
	anchors: ReadonlySet<string>,
	width: number | undefined,
): string {
	const output: string[] = [];
	let sourceLines: string[] = [];

	const flushSourceLines = () => {
		if (sourceLines.length === 0) return;
		output.push(...wrapDisplayLines(renderSourceBlock(sourceLines, path), width));
		sourceLines = [];
	};

	for (const line of text.split("\n")) {
		const match = line.match(READ_ANCHOR_RE);
		if (match && anchors.has(match[1]!)) {
			sourceLines.push(match[2] ?? "");
			continue;
		}
		flushSourceLines();
		output.push(...wrapDisplayLines([line], width));
	}
	flushSourceLines();
	return output.join("\n");
}

const readSourceCache = new WeakMap<object, {
	text: string;
	path: string;
	width: number | undefined;
	theme: string;
	anchors: string;
	rendered: string;
}>();

export function renderReadSourceForDisplayCached(
	cacheKey: object,
	text: string,
	path: string,
	anchors: ReadonlySet<string>,
	width: number | undefined,
	theme: SourceRenderTheme,
): string {
	const themeKey = SOURCE_THEME_STYLES.map((style) => theme.fg(style, "")).join("");
	const anchorKey = [...anchors].join("\n");
	const cached = readSourceCache.get(cacheKey);
	if (
		cached &&
		cached.text === text &&
		cached.path === path &&
		cached.width === width &&
		cached.theme === themeKey &&
		cached.anchors === anchorKey
	) {
		return cached.rendered;
	}
	const rendered = renderReadSourceForDisplay(text, path, anchors, width);
	readSourceCache.set(cacheKey, { text, path, width, theme: themeKey, anchors: anchorKey, rendered });
	return rendered;
}

export function renderGrepSourceForDisplay(
	text: string,
	anchors: ReadonlySet<string>,
	renderPath: (path: string) => string = (path) => path,
): string {
	return text
		.split("\n")
		.map((line) => {
			const match = line.match(GREP_ANCHOR_RE);
			if (!match) return line;
			const anchor = `${match[3]}:${match[4]}`;
			if (!anchors.has(anchor)) return line;
			const separator = match[2] === ">>" ? ":" : "-";
			return `${renderPath(match[1]!)}${separator}${match[3]}${separator}${match[5] ?? ""}`;
		})
		.join("\n");
}
