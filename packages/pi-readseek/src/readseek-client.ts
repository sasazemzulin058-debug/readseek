import { spawn, type StdioOptions } from "node:child_process";
import { existsSync, readFileSync, statSync } from "node:fs";
import { createRequire } from "node:module";
import { homedir } from "node:os";
import path from "node:path";

import { DetailLevel } from "./readseek/enums.js";
import type { FileMap, FileSymbol } from "./readseek/types.js";
import { SymbolKind } from "./readseek/enums.js";
import { resolveReadSeekTimeoutMs } from "./readseek-settings.js";

export interface ReadSeekHashline {
	line: number;
	hash: string;
	text: string;
}

interface ReadSeekReadOutput {
	file: string;
	language: string;
	line_count: number;
	file_hash: string;
	start_line: number;
	end_line: number;
	hashlines: ReadSeekHashline[];
}

interface ReadSeekSymbol {
	kind: string;
	name: string;
	qualified_name: string;
	start_line: number;
	end_line: number;
	start_hash: string;
	end_hash: string;
}

interface ReadSeekMapOutput {
	file: string;
	language: string;
	line_count: number;
	file_hash: string;
	symbols: ReadSeekSymbol[];
}

interface ReadSeekSearchCapture {
	name: string;
	start_line: number;
	end_line: number;
	start_hash: string;
	end_hash: string;
	hashlines: ReadSeekHashline[];
}

interface ReadSeekSearchMatch {
	pattern_index: number;
	start_line: number;
	end_line: number;
	start_hash: string;
	end_hash: string;
	hashlines: ReadSeekHashline[];
	captures: ReadSeekSearchCapture[];
}

export interface ReadSeekSearchFileOutput {
	file: string;
	language: string;
	file_hash: string;
	matches: ReadSeekSearchMatch[];
}

interface ReadSeekSearchOutput {
	results: ReadSeekSearchFileOutput[];
}

export interface ReadSeekReference {
	file: string;
	line: number;
	column: number;
	line_hash: string;
	text: string;
	enclosingSymbol?: string;
}

interface ReadSeekRefsOutput {
	references: ReadSeekReference[];
}

interface ReadSeekRefsOptions {
	scope?: boolean;
	line?: number;
	column?: number;
	language?: string;
	cached?: boolean;
	others?: boolean;
	ignored?: boolean;
	signal?: AbortSignal;
}

export interface ReadSeekDiagnostic {
	kind: "error" | "missing";
	start_line: number;
	end_line: number;
}

export interface ReadSeekCheckOutput {
	errorCount: number;
	missingCount: number;
	diagnostics: ReadSeekDiagnostic[];
}

export type ReadSeekOcrText = string;

export interface ReadSeekDetectedObject {
	label: string;
	bbox: [number, number, number, number];
}

export type ReadSeekImageMode = "all" | "ocr" | "caption" | "objects";

export interface ReadSeekPreparedImage {
	mime: string;
	encoding: "base64";
	data: string;
}

export interface ReadSeekPdfImage {
	page: number;
	width: number;
	height: number;
	mime: string;
	mode: "none" | ReadSeekImageMode;
	encoding?: "base64";
	data?: string;
	ocr?: ReadSeekOcrText;
	caption?: string;
	objects?: ReadSeekDetectedObject[];
}

export interface ReadSeekPdfOutput {
	format: "pdf";
	pages: number;
	markdown: string;
	images: ReadSeekPdfImage[];
}

export type ReadSeekDetection =
	| {
			kind: "source";
			type: string;
			file: string;
			language: string;
			engine?: string;
			supported: boolean;
			mime?: string;
			syntax?: string;
		}
	| {
			kind: "image";
			type: string;
			file: string;
			mime?: string;
			format: string;
			width: number;
			height: number;
			animated: boolean;
			encoding?: "base64";
			data?: string;
			ocr?: ReadSeekOcrText;
			caption?: string;
			objects?: ReadSeekDetectedObject[];
		}
	| {
			kind: "text";
			type: string;
			file: string;
			mime?: string;
		}
	| {
			kind: "pdf";
			type: "application/pdf";
			file: string;
			mime?: string;
			format: "pdf";
			pages: number;
		};

type ReadSeekImageDetection = Extract<ReadSeekDetection, { kind: "image" }>;

interface ReadSeekSearchOptions {
	language?: string;
	cached?: boolean;
	others?: boolean;
	ignored?: boolean;
	signal?: AbortSignal;
}

function normalizeLanguage(language: string): string {
	return language === "java" ? "Java" : language;
}

function normalizeKind(kind: string): FileSymbol["kind"] {
	if (kind === "constructor") return SymbolKind.Method;
	if (Object.values(SymbolKind).includes(kind as SymbolKind)) return kind as FileSymbol["kind"];
	return SymbolKind.Unknown;
}

function parentQualifiedNameFor(qualifiedName: string): string {
	const lastDot = qualifiedName.lastIndexOf(".");
	return lastDot === -1 ? "" : qualifiedName.slice(0, lastDot);
}

function symbolsFromReadSeek(symbols: ReadSeekSymbol[]): FileSymbol[] {
	const symbolsByQualifiedName = new Map<string, FileSymbol[]>();
	const entries: Array<{ parentQualifiedName: string; symbol: FileSymbol }> = [];

	for (const symbol of symbols) {
		const parentQualifiedName = parentQualifiedNameFor(symbol.qualified_name);
		const fileSymbol: FileSymbol = {
			name: symbol.name,
			kind: normalizeKind(symbol.kind),
			startLine: symbol.start_line,
			endLine: symbol.end_line,
		};
		const bucket = symbolsByQualifiedName.get(symbol.qualified_name);
		if (bucket) bucket.push(fileSymbol);
		else symbolsByQualifiedName.set(symbol.qualified_name, [fileSymbol]);
		entries.push({ parentQualifiedName, symbol: fileSymbol });
	}

	const roots: FileSymbol[] = [];
	for (const entry of entries) {
		const parent = entry.parentQualifiedName
			? symbolsByQualifiedName.get(entry.parentQualifiedName)?.[0]
			: undefined;
		if (!parent) {
			roots.push(entry.symbol);
			continue;
		}

		parent.children ??= [];
		parent.children.push(entry.symbol);
	}

	return roots;
}

const require = createRequire(import.meta.url);
const READSEEK_REPO_PACKAGE_NAMES = new Set(["@jarkkojs/readseek", "readseek"]);
let defaultReadSeekDirInit: Promise<string | null> | undefined;

function readSeekPackageDir(): string {
	return path.dirname(require.resolve("@jarkkojs/readseek/package.json"));
}

const READSEEK_PLATFORM_PACKAGES: Record<string, string> = {
	"android-arm64": "@sasazemzulin058-debug/readseek-android-arm64",
	"darwin-arm64": "@jarkkojs/readseek-darwin-arm64",
	"linux-arm64": "@jarkkojs/readseek-linux-arm64",
	"linux-x64": "@jarkkojs/readseek-linux-x64",
	"win32-x64": "@jarkkojs/readseek-win32-x64",
};

function readSeekPlatform(): string {
	return `${process.platform}-${process.arch}`;
}

function readSeekBinaryPath(): string {
	const platform = readSeekPlatform();
	const platformPackage = READSEEK_PLATFORM_PACKAGES[platform];
	if (!platformPackage) {
		const supported = Object.keys(READSEEK_PLATFORM_PACKAGES).join(", ");
		throw new Error(`@jarkkojs/readseek ships no binary for ${platform}; it supports ${supported}`);
	}

	const packageJson = require.resolve(`${platformPackage}/package.json`, { paths: [readSeekPackageDir()] });
	return path.join(path.dirname(packageJson), "bin", process.platform === "win32" ? "readseek.exe" : "readseek");
}

/**
 * Report whether a readseek binary can be resolved for the running platform,
 * along with the reason when it cannot. Used to keep the readseek tools out of
 * the active set on hosts that readseek publishes no binary for.
 */
export function readSeekBinaryAvailability(): { available: true } | { available: false; reason: string } {
	try {
		readSeekBinaryPath();
		return { available: true };
	} catch (err) {
		return { available: false, reason: classifyReadSeekFailure(err).message };
	}
}

interface ReadSeekFailure {
	code: "readseek-not-installed" | "readseek-execution-error";
	message: string;
	hint?: string;
}

/**
 * Classify an error thrown while invoking readseek into the shared failure
 * taxonomy: a missing binary or package (`readseek-not-installed`, with an
 * install hint) versus any other execution error.
 */
export function classifyReadSeekFailure(err: unknown): ReadSeekFailure {
	const failure = err as { code?: unknown; message?: unknown } | null;
	const message = String(failure?.message ?? err);
	const missing =
		failure?.code === "ENOENT" ||
		/Cannot find package|Cannot find module|no such file/i.test(message);
	if (missing) {
		return { code: "readseek-not-installed", message, hint: "Run npm install to install @jarkkojs/readseek." };
	}
	return { code: "readseek-execution-error", message };
}

function directoryExists(dirPath: string): boolean {
	try {
		return statSync(dirPath).isDirectory();
	} catch {
		return false;
	}
}

const ownRepositoryByDir = new Map<string, boolean>();

function isOwnReadSeekRepository(cwd = process.cwd()): boolean {
	const start = path.resolve(cwd);
	const cached = ownRepositoryByDir.get(start);
	if (cached !== undefined) return cached;

	let dir = start;
	let result = false;
	while (true) {
		const packageJsonPath = path.join(dir, "package.json");
		if (existsSync(packageJsonPath)) {
			try {
				const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8")) as { name?: unknown };
				if (typeof packageJson.name === "string" && READSEEK_REPO_PACKAGE_NAMES.has(packageJson.name)) {
					result = true;
					break;
				}
			} catch {
				// Ignore unreadable or invalid package manifests while walking up.
			}
		}

		const parent = path.dirname(dir);
		if (parent === dir) break;
		dir = parent;
	}

	ownRepositoryByDir.set(start, result);
	return result;
}

function defaultReadSeekDir(): string | null {
	const home = homedir();
	return home ? path.join(home, ".pi", "readseek") : null;
}

const DEFAULT_READSEEK_TIMEOUT_MS = 120_000;
const DEFAULT_READSEEK_VISION_TIMEOUT_MS = 30 * 60_000;

function readSeekTimeoutMs(): number {
	return resolveReadSeekTimeoutMs() ?? DEFAULT_READSEEK_TIMEOUT_MS;
}

const READSEEK_USAGE_HINT = /\n\s*Run readseek(?: [\w-]+)* --help for more information\.?\s*$/;

function readSeekErrorMessage(stderr: string): string {
	return stderr
		.replace(READSEEK_USAGE_HINT, "")
		.replace(/^error:\s*/i, "")
		.replace(/\s*\n\s*/g, " ")
		.trim();
}

async function spawnReadSeekRaw(args: string[], options: RunReadSeekOptions = {}): Promise<string> {
	return new Promise<string>((resolve, reject) => {
		let settled = false;
		let timeout: ReturnType<typeof setTimeout> | undefined;
		const fail = (error: Error): void => {
			if (settled) return;
			settled = true;
			if (timeout !== undefined) clearTimeout(timeout);
			reject(error);
		};
		const succeed = (value: string): void => {
			if (settled) return;
			settled = true;
			if (timeout !== undefined) clearTimeout(timeout);
			resolve(value);
		};

		const stdin = options.stdin;
		const stdio: StdioOptions = [stdin === undefined ? "ignore" : "pipe", "pipe", "pipe"];
		const child = spawn(readSeekBinaryPath(), args, { stdio, signal: options.signal });
		const timeoutMs = options.timeoutMs ?? readSeekTimeoutMs();
		timeout = setTimeout(() => {
			child.kill("SIGKILL");
			fail(new Error(`readseek timed out after ${timeoutMs} ms`));
		}, timeoutMs);
		timeout.unref?.();
		const childStdout = child.stdout;
		const childStderr = child.stderr;
		const childStdin = child.stdin;
		if (!childStdout || !childStderr) {
			child.kill();
			fail(new Error("readseek stdio streams are unavailable"));
			return;
		}
		const stdoutChunks: Buffer[] = [];
		const stderrChunks: Buffer[] = [];
		let stdoutBytes = 0;

		childStdout.on("data", (chunk: Buffer) => {
			if (settled) return;
			stdoutBytes += chunk.length;
			if (stdoutBytes > 32 * 1024 * 1024) {
				child.kill();
				fail(new Error("readseek output exceeded 32 MiB"));
				return;
			}
			stdoutChunks.push(chunk);
		});
		childStderr.on("data", (chunk: Buffer) => stderrChunks.push(chunk));
		child.on("error", (error: any) => fail(error));
		if (stdin !== undefined) {
			if (!childStdin) {
				child.kill();
				fail(new Error("readseek stdin stream is unavailable"));
				return;
			}
			childStdin.on("error", (error: any) => {
				if (error?.code !== "EPIPE") fail(error);
			});
			childStdin.end(stdin, "utf-8");
		}
		child.on("close", (code, signal) => {
			const stdout = Buffer.concat(stdoutChunks).toString("utf-8");
			const stderr = Buffer.concat(stderrChunks).toString("utf-8").trim();
			if (code === 0) succeed(stdout);
			else if (signal) fail(new Error(readSeekErrorMessage(stderr) || `readseek killed by signal ${signal}`));
			else fail(new Error(readSeekErrorMessage(stderr) || `readseek exited with status ${code}`));
		});
	});
}

async function ensureDefaultReadSeekDir(): Promise<string | null> {
	const dir = defaultReadSeekDir();
	if (!dir) return null;
	if (directoryExists(dir)) return dir;

	defaultReadSeekDirInit ??= spawnReadSeekRaw(["--readseek-dir", dir, "init"])
		.then(() => (directoryExists(dir) ? dir : null))
		.catch(() => null)
		.finally(() => {
			defaultReadSeekDirInit = undefined;
		});
	return defaultReadSeekDirInit;
}

async function readSeekInvocationArgs(args: string[]): Promise<string[]> {
	if (isOwnReadSeekRepository()) return args;

	const readSeekDir = await ensureDefaultReadSeekDir();
	return readSeekDir ? ["--readseek-dir", readSeekDir, ...args] : args;
}

interface RunReadSeekOptions {
	signal?: AbortSignal;
	stdin?: string;
	timeoutMs?: number;
}

async function runReadSeekRaw(args: string[], options: RunReadSeekOptions = {}): Promise<string> {
	return spawnReadSeekRaw(await readSeekInvocationArgs(args), options);
}

async function runReadSeek(args: string[], options: RunReadSeekOptions = {}): Promise<unknown> {
	const stdout = await runReadSeekRaw(args, options);
	return JSON.parse(stdout) as unknown;
}

export interface ReadSeekViewOptions {
	node?: string;
	page?: number;
	kind?: string;
	depth?: number;
	outline?: boolean;
	signal?: AbortSignal;
}

export async function readSeekView(filePath: string, options: ReadSeekViewOptions = {}): Promise<string> {
	const args = ["view", filePath];
	if (options.node !== undefined) args.push("--node", options.node);
	if (options.page !== undefined) args.push("--page", String(options.page));
	if (options.kind !== undefined) args.push("--kind", options.kind);
	if (options.depth !== undefined) args.push("--depth", String(options.depth));
	if (options.outline) args.push("--outline");
	return runReadSeekRaw(args, { signal: options.signal });
}

let visionInvocationTail = Promise.resolve();

/**
 * Serialize local vision-model processes. Concurrent processes each consume all
 * available CPU cores, which makes every invocation exceed its own timeout.
 */
async function runReadSeekVision(args: string[], options: RunReadSeekOptions = {}): Promise<unknown> {
	let release!: () => void;
	const gate = new Promise<void>((resolve) => {
		release = resolve;
	});
	const predecessor = visionInvocationTail;
	visionInvocationTail = predecessor.then(() => gate);
	const signal = options.signal;

	try {
		if (signal) {
			signal.throwIfAborted();
			await new Promise<void>((resolve, reject) => {
				const onAbort = (): void => {
					signal.removeEventListener("abort", onAbort);
					reject(signal.reason);
				};
				signal.addEventListener("abort", onAbort, { once: true });
				predecessor.then(
					() => {
						signal.removeEventListener("abort", onAbort);
						resolve();
					},
					(error) => {
						signal.removeEventListener("abort", onAbort);
						reject(error);
					},
				);
			});
		} else {
			await predecessor;
		}
		signal?.throwIfAborted();
		return await runReadSeek(args, {
			...options,
			timeoutMs: options.timeoutMs ?? resolveReadSeekTimeoutMs() ?? DEFAULT_READSEEK_VISION_TIMEOUT_MS,
		});
	} finally {
		release();
	}
}

function requireNumber(value: unknown, field: string): number {
	if (typeof value !== "number" || !Number.isSafeInteger(value)) throw new Error(`invalid readseek ${field}: expected safe integer`);
	return value;
}

function requireString(value: unknown, field: string): string {
	if (typeof value !== "string") throw new Error(`invalid readseek ${field}`);
	return value;
}

function requireBoolean(value: unknown, field: string): boolean {
	if (typeof value !== "boolean") throw new Error(`invalid readseek ${field}: expected boolean`);
	return value;
}

function parseReadOutput(value: unknown): ReadSeekReadOutput {
	if (!value || typeof value !== "object") throw new Error("invalid readseek read output");
	const output = value as Record<string, unknown>;
	const hashlines = output.hashlines;
	if (!Array.isArray(hashlines)) throw new Error("invalid readseek hashlines");
	return {
		file: requireString(output.file, "file"),
		language: requireString(output.language, "language"),
		line_count: requireNumber(output.line_count, "line_count"),
		file_hash: requireString(output.file_hash, "file_hash"),
		start_line: requireNumber(output.start_line, "start_line"),
		end_line: requireNumber(output.end_line, "end_line"),
		hashlines: hashlines.map((line) => parseHashline(line, "hashline")),
	};
}

function parseMapOutput(value: unknown): ReadSeekMapOutput {
	if (!value || typeof value !== "object") throw new Error("invalid readseek map output");
	const output = value as Record<string, unknown>;
	const symbols = output.symbols;
	if (!Array.isArray(symbols)) throw new Error("invalid readseek symbols");
	return {
		file: requireString(output.file, "file"),
		language: requireString(output.language, "language"),
		line_count: requireNumber(output.line_count, "line_count"),
		file_hash: requireString(output.file_hash, "file_hash"),
		symbols: symbols.map((symbol) => {
			if (!symbol || typeof symbol !== "object") throw new Error("invalid readseek symbol");
			const item = symbol as Record<string, unknown>;
			return {
				kind: requireString(item.kind, "symbol.kind"),
				name: requireString(item.name, "symbol.name"),
				qualified_name: requireString(item.qualified_name, "symbol.qualified_name"),
				start_line: requireNumber(item.start_line, "symbol.start_line"),
				end_line: requireNumber(item.end_line, "symbol.end_line"),
				start_hash: requireString(item.start_hash, "symbol.start_hash"),
				end_hash: requireString(item.end_hash, "symbol.end_hash"),
			};
		}),
	};
}

function parseHashline(value: unknown, field: string): ReadSeekHashline {
	if (!value || typeof value !== "object") throw new Error(`invalid readseek ${field}`);
	const item = value as Record<string, unknown>;
	return {
		line: requireNumber(item.line, `${field}.line`),
		hash: requireString(item.hash, `${field}.hash`),
		text: requireString(item.text, `${field}.text`),
	};
}

function parseSearchHashlines(value: unknown, field: string): ReadSeekHashline[] {
	if (!Array.isArray(value)) throw new Error(`invalid readseek ${field}`);
	return value.map((line) => parseHashline(line, field));
}

function parseSearchOutput(value: unknown): ReadSeekSearchOutput {
	if (!value || typeof value !== "object") throw new Error("invalid readseek search output");
	const output = value as Record<string, unknown>;
	if (!Array.isArray(output.results)) throw new Error("invalid readseek search results");
	return {
		results: output.results.map((result) => {
			if (!result || typeof result !== "object") throw new Error("invalid readseek search result");
			const file = result as Record<string, unknown>;
			if (!Array.isArray(file.matches)) throw new Error("invalid readseek search matches");
			return {
				file: requireString(file.file, "search.file"),
				language: requireString(file.language, "search.language"),
				file_hash: requireString(file.file_hash, "search.file_hash"),
				matches: file.matches.map((match) => {
					if (!match || typeof match !== "object") throw new Error("invalid readseek search match");
					const item = match as Record<string, unknown>;
					if (!Array.isArray(item.captures)) throw new Error("invalid readseek search captures");
					return {
						pattern_index: item.pattern_index === undefined ? 0 : requireNumber(item.pattern_index, "search.match.pattern_index"),
						start_line: requireNumber(item.start_line, "search.match.start_line"),
						end_line: requireNumber(item.end_line, "search.match.end_line"),
						start_hash: requireString(item.start_hash, "search.match.start_hash"),
						end_hash: requireString(item.end_hash, "search.match.end_hash"),
						hashlines: parseSearchHashlines(item.hashlines, "search.match.hashlines"),
						captures: item.captures.map((capture) => {
							if (!capture || typeof capture !== "object") throw new Error("invalid readseek search capture");
							const captureItem = capture as Record<string, unknown>;
							return {
								name: requireString(captureItem.name, "search.capture.name"),
								start_line: requireNumber(captureItem.start_line, "search.capture.start_line"),
								end_line: requireNumber(captureItem.end_line, "search.capture.end_line"),
								start_hash: requireString(captureItem.start_hash, "search.capture.start_hash"),
								end_hash: requireString(captureItem.end_hash, "search.capture.end_hash"),
								hashlines: parseSearchHashlines(captureItem.hashlines, "search.capture.hashlines"),
							};
						}),
					};
				}),
			};
		}),
	};
}

export async function readSeekRead(
	filePath: string,
	startLine?: number,
	endLine?: number,
	options: { signal?: AbortSignal } = {},
): Promise<ReadSeekReadOutput> {
	const args = ["read", startLine === undefined ? filePath : `${filePath}:${startLine}`];
	if (endLine !== undefined) args.push("--end", String(endLine));
	return parseReadOutput(await runReadSeek(args, { signal: options.signal }));
}

function fileMapFromReadSeekOutput(output: ReadSeekMapOutput, filePath: string, totalBytes: number): FileMap | null {
	if (output.language === "unknown" && output.symbols.length === 0) return null;
	return {
		path: filePath,
		totalLines: output.line_count,
		totalBytes,
		language: normalizeLanguage(output.language),
		detailLevel: DetailLevel.Full,
		symbols: symbolsFromReadSeek(output.symbols),
	};
}

export async function readSeekMap(
	filePath: string,
	totalBytes: number,
	options: { signal?: AbortSignal } = {},
): Promise<FileMap | null> {
	const output = parseMapOutput(await runReadSeek(["map", filePath], { signal: options.signal }));
	return fileMapFromReadSeekOutput(output, filePath, totalBytes);
}

export async function readSeekSearch(
	target: string,
	pattern: string,
	options: ReadSeekSearchOptions = {},
): Promise<ReadSeekSearchFileOutput[]> {
	const args = ["search", target, pattern];
	if (options.language) args.push("--language", options.language);
	if (options.cached) args.push("--cached");
	if (options.others) args.push("--others");
	if (options.ignored) args.push("--ignored");
	return parseSearchOutput(await runReadSeek(args, { signal: options.signal })).results;
}

export async function readSeekMapContent(
	filePath: string,
	content: string,
	options: { signal?: AbortSignal } = {},
): Promise<FileMap | null> {
	const output = parseMapOutput(
		await runReadSeek(["map", `stdin:${filePath}`], { signal: options.signal, stdin: content }),
	);
	return fileMapFromReadSeekOutput(output, filePath, Buffer.byteLength(content, "utf8"));
}

function optionalString(value: unknown, field: string): string | undefined {
	if (value === undefined || value === null) return undefined;
	return requireString(value, field);
}

function parseRefsOutput(value: unknown): ReadSeekRefsOutput {
	if (!value || typeof value !== "object") throw new Error("invalid readseek refs output");
	const output = value as Record<string, unknown>;
	if (!Array.isArray(output.references)) throw new Error("invalid readseek references");
	return {
		references: output.references.map((reference) => {
			if (!reference || typeof reference !== "object") throw new Error("invalid readseek reference");
			const item = reference as Record<string, unknown>;
			const symbol = item.symbol;
			const enclosing =
				symbol && typeof symbol === "object"
					? optionalString((symbol as Record<string, unknown>).qualified_name, "reference.symbol.qualified_name")
					: undefined;
			return {
				file: requireString(item.file, "reference.file"),
				line: requireNumber(item.line, "reference.line"),
				column: requireNumber(item.column, "reference.column"),
				line_hash: requireString(item.line_hash, "reference.line_hash"),
				text: requireString(item.text, "reference.text"),
				enclosingSymbol: enclosing,
			};
		}),
	};
}

export async function readSeekRefs(
	target: string,
	name: string,
	options: ReadSeekRefsOptions = {},
): Promise<ReadSeekReference[]> {
	const args = ["refs", target, name];
	if (options.scope) args.push("--scope");
	if (options.line !== undefined) args.push("--line", String(options.line));
	if (options.column !== undefined) args.push("--column", String(options.column));
	if (options.language) args.push("--language", options.language);
	if (options.cached) args.push("--cached");
	if (options.others) args.push("--others");
	if (options.ignored) args.push("--ignored");
	return parseRefsOutput(await runReadSeek(args, { signal: options.signal })).references;
}

function parseDiagnosticKind(value: unknown): ReadSeekDiagnostic["kind"] {
	if (value === "error" || value === "missing") return value;
	throw new Error("invalid readseek diagnostic.kind");
}

function parseCheckOutput(value: unknown): ReadSeekCheckOutput {
	if (!value || typeof value !== "object") throw new Error("invalid readseek check output");
	const output = value as Record<string, unknown>;
	if (!Array.isArray(output.diagnostics)) throw new Error("invalid readseek diagnostics");
	return {
		errorCount: requireNumber(output.error_count, "error_count"),
		missingCount: requireNumber(output.missing_count, "missing_count"),
		diagnostics: output.diagnostics.map((diagnostic) => {
			if (!diagnostic || typeof diagnostic !== "object") throw new Error("invalid readseek diagnostic");
			const item = diagnostic as Record<string, unknown>;
			return {
				kind: parseDiagnosticKind(item.kind),
				start_line: requireNumber(item.start_line, "diagnostic.start_line"),
				end_line: requireNumber(item.end_line, "diagnostic.end_line"),
			};
		}),
	};
}

export async function readSeekCheck(
	filePath: string,
	content: string,
	options: { signal?: AbortSignal } = {},
): Promise<ReadSeekCheckOutput> {
	return parseCheckOutput(
		await runReadSeek(["check", `stdin:${filePath}`], { signal: options.signal, stdin: content }),
	);
}

function parseOcrText(value: unknown): ReadSeekOcrText | undefined {
	return optionalString(value, "ocr");
}

function parseDetectedObjects(value: unknown): ReadSeekDetectedObject[] | undefined {
	if (value === undefined || value === null) return undefined;
	if (!Array.isArray(value)) throw new Error("invalid readseek detect objects");
	return value.map((object) => {
		if (!object || typeof object !== "object") throw new Error("invalid readseek detect object");
		const item = object as Record<string, unknown>;
		const bbox = item.bbox;
		if (!Array.isArray(bbox) || bbox.length !== 4) throw new Error("invalid readseek detect object.bbox");
		return {
			label: requireString(item.label, "object.label"),
			bbox: bbox.map((n, i) => requireNumber(n, `object.bbox[${i}]`)) as ReadSeekDetectedObject["bbox"],
		};
	});
}

function parseDetectOutput(value: unknown): ReadSeekDetection {
	if (!value || typeof value !== "object") throw new Error("invalid readseek detect output");
	const output = value as Record<string, unknown>;
	const type = requireString(output.type, "type");
	const file = requireString(output.file, "file");
	const mime = optionalString(output.mime, "mime");

	// readseek 0.4.22 made the detect enum untagged and repurposed `type` to
	// carry the actual MIME type. Discriminate structurally: image detections
	// carry `format`/`width`/`height`/`animated`; source detections carry
	// `language`. Text and binary are byte-identical on the wire and collapse
	// to the text variant.
	if (type === "application/pdf") {
		if (output.format !== "pdf") throw new Error("invalid readseek PDF format");
		return {
			kind: "pdf",
			type,
			file,
			mime,
			format: output.format,
			pages: requireNumber(output.pages, "pages"),
		};
	}
	if (output.width !== undefined || output.height !== undefined) {
		return {
			kind: "image",
			type,
			file,
			mime,
			format: requireString(output.format, "format"),
			width: requireNumber(output.width, "width"),
			height: requireNumber(output.height, "height"),
			animated: requireBoolean(output.animated, "animated"),
			encoding: output.encoding === undefined ? undefined : parseImageEncoding(output.encoding),
			data: output.data === undefined ? undefined : requireString(output.data, "data"),
			ocr: parseOcrText(output.ocr),
			caption: optionalString(output.caption, "caption"),
			objects: parseDetectedObjects(output.objects),
		};
	}
	if (output.language !== undefined) {
		return {
			kind: "source",
			type,
			file,
			language: requireString(output.language, "language"),
			engine: optionalString(output.engine, "engine"),
			supported: requireBoolean(output.supported, "supported"),
			mime,
			syntax: optionalString(output.syntax, "syntax"),
		};
	}
	return { kind: "text", type, file, mime };
}

function parseImageEncoding(value: unknown): "base64" {
	if (value === "base64") return value;
	throw new Error("invalid image encoding");
}

export async function readSeekDetect(
	filePath: string,
	options: { signal?: AbortSignal } = {},
): Promise<ReadSeekDetection> {
	return parseDetectOutput(await runReadSeek(["detect", filePath], { signal: options.signal }));
}

/**
 * Analyze an image with each requested vision mode and merge the payloads into
 * a single detection. readseek accepts one `--vision-mode` per invocation, so
 * modes that fail are dropped as long as at least one produced a payload;
 * otherwise the first failure is rethrown.
 */
export async function readSeekImage(
	filePath: string,
	modes: ReadSeekImageMode[],
	options: { signal?: AbortSignal } = {},
): Promise<ReadSeekDetection> {
	const requestedModes = modes.length > 1 ? ["all"] : modes;
	const results = await Promise.allSettled(
		requestedModes.map(async (mode) =>
			parseDetectOutput(await runReadSeekVision(["read", "--vision-mode", mode, filePath], { signal: options.signal })),
		),
	);

	let merged: ReadSeekImageDetection | undefined;
	for (const result of results) {
		if (result.status !== "fulfilled" || result.value.kind !== "image") continue;
		const detection = result.value;
		merged = merged === undefined
			? detection
			: {
				...merged,
				ocr: detection.ocr ?? merged.ocr,
				caption: detection.caption ?? merged.caption,
				objects: detection.objects ?? merged.objects,
			};
	}
	if (merged !== undefined) return merged;

	const failure = results.find((result) => result.status === "rejected");
	if (failure?.status === "rejected") throw failure.reason;
	throw new Error(`readseek returned no image analysis for ${filePath}`);
}

export async function readSeekPreparedImage(
	filePath: string,
	options: { signal?: AbortSignal } = {},
): Promise<ReadSeekPreparedImage> {
	const output = parseDetectOutput(await runReadSeek(["read", "--vision-mode", "none", filePath], { signal: options.signal }));
	if (output.kind !== "image" || output.encoding !== "base64" || output.data === undefined || output.mime === undefined) {
		throw new Error(`readseek returned no prepared image for ${filePath}`);
	}
	return { mime: output.mime, encoding: output.encoding, data: output.data };
}

function parsePdfImage(value: unknown): ReadSeekPdfImage {
	if (!value || typeof value !== "object") throw new Error("invalid readseek PDF image");
	const image = value as Record<string, unknown>;
	const mode = requireString(image.mode, "PDF image.mode");
	if (mode !== "none" && mode !== "all" && mode !== "ocr" && mode !== "caption" && mode !== "objects") {
		throw new Error("invalid readseek PDF image.mode");
	}
	return {
		page: requireNumber(image.page, "PDF image.page"),
		width: requireNumber(image.width, "PDF image.width"),
		height: requireNumber(image.height, "PDF image.height"),
		mime: requireString(image.mime, "PDF image.mime"),
		mode,
		encoding: image.encoding === undefined ? undefined : parseImageEncoding(image.encoding),
		data: optionalString(image.data, "PDF image.data"),
		ocr: parseOcrText(image.ocr),
		caption: optionalString(image.caption, "PDF image.caption"),
		objects: parseDetectedObjects(image.objects),
	};
}

function parsePdfOutput(value: unknown): ReadSeekPdfOutput {
	if (!value || typeof value !== "object") throw new Error("invalid readseek PDF output");
	const output = value as Record<string, unknown>;
	if (output.format !== "pdf" || !Array.isArray(output.images)) throw new Error("invalid readseek PDF output");
	return {
		format: output.format,
		pages: requireNumber(output.pages, "PDF pages"),
		markdown: requireString(output.markdown, "PDF markdown"),
		images: output.images.map(parsePdfImage),
	};
}

export async function readSeekPdf(
	filePath: string,
	mode: "none" | ReadSeekImageMode,
	options: { page?: number; signal?: AbortSignal } = {},
): Promise<ReadSeekPdfOutput> {
	const run = mode === "none" ? runReadSeek : runReadSeekVision;
	const args = ["read", "--vision-mode", mode, filePath];
	if (options.page !== undefined) args.push("--page", String(options.page));
	return parsePdfOutput(await run(args, { signal: options.signal }));
}

// --- Rename ---

export interface RenameConflict {
	line: number;
	column: number;
	reason: string;
}

export interface RenameEdit {
	line: number;
	start_column: number;
	end_column: number;
	start_byte: number;
	end_byte: number;
	occurrence: string;
	line_hash: string;
	text: string;
}

export interface RenameFileOutput {
	file: string;
	language: string;
	engine?: string;
	file_hash: string;
	conflicts: RenameConflict[];
	edits: RenameEdit[];
}

export interface RenameOutput {
	file: string;
	language: string;
	engine?: string;
	file_hash: string;
	old_name: string;
	new_name: string;
	applied: boolean;
	conflicts: RenameConflict[];
	edits: RenameEdit[];
	others: RenameFileOutput[];
}

interface RenameOptions {
	to: string;
	line: number;
	column?: number;
	workspace?: string;
	apply?: boolean;
	language?: string;
	cached?: boolean;
	others?: boolean;
	ignored?: boolean;
	signal?: AbortSignal;
}

function parseRenameConflicts(value: unknown, field: string): RenameConflict[] {
	if (value === undefined) return [];
	if (!Array.isArray(value)) throw new Error(`invalid readseek ${field}`);
	return value.map((item) => {
		if (!item || typeof item !== "object") throw new Error(`invalid readseek ${field} entry`);
		const c = item as Record<string, unknown>;
		return {
			line: requireNumber(c.line, `${field}.line`),
			column: requireNumber(c.column, `${field}.column`),
			reason: requireString(c.reason, `${field}.reason`),
		};
	});
}

function parseRenameEdits(value: unknown, field: string): RenameEdit[] {
	if (value === undefined) return [];
	if (!Array.isArray(value)) throw new Error(`invalid readseek ${field}`);
	return value.map((item) => {
		if (!item || typeof item !== "object") throw new Error(`invalid readseek ${field} entry`);
		const e = item as Record<string, unknown>;
		return {
			line: requireNumber(e.line, `${field}.line`),
			start_column: requireNumber(e.start_column, `${field}.start_column`),
			end_column: requireNumber(e.end_column, `${field}.end_column`),
			start_byte: requireNumber(e.start_byte, `${field}.start_byte`),
			end_byte: requireNumber(e.end_byte, `${field}.end_byte`),
			occurrence: requireString(e.occurrence, `${field}.occurrence`),
			line_hash: requireString(e.line_hash, `${field}.line_hash`),
			text: requireString(e.text, `${field}.text`),
		};
	});
}

function parseRenameOutput(value: unknown): RenameOutput {
	if (!value || typeof value !== "object") throw new Error("invalid readseek rename output");
	const output = value as Record<string, unknown>;
	const others = output.others;
	if (others !== undefined && !Array.isArray(others)) throw new Error("invalid readseek others");
	return {
		file: requireString(output.file, "file"),
		language: requireString(output.language, "language"),
		engine: optionalString(output.engine, "engine"),
		file_hash: requireString(output.file_hash, "file_hash"),
		old_name: requireString(output.old_name, "old_name"),
		new_name: requireString(output.new_name, "new_name"),
		applied: requireBoolean(output.applied, "applied"),
		conflicts: parseRenameConflicts(output.conflicts, "conflicts"),
		edits: parseRenameEdits(output.edits, "edits"),
		others: (others as unknown[] | undefined)?.map((entry) => {
			if (!entry || typeof entry !== "object") throw new Error("invalid readseek other");
			const o = entry as Record<string, unknown>;
			return {
				file: requireString(o.file, "other.file"),
				language: requireString(o.language, "other.language"),
				engine: optionalString(o.engine, "other.engine"),
				file_hash: requireString(o.file_hash, "other.file_hash"),
				conflicts: parseRenameConflicts(o.conflicts, "other.conflicts"),
				edits: parseRenameEdits(o.edits, "other.edits"),
			};
		}) ?? [],
	};
}

export async function readSeekRename(
	filePath: string,
	options: RenameOptions,
): Promise<RenameOutput> {
	const args = ["rename", filePath, "--line", String(options.line)];
	if (options.column !== undefined) args.push("--column", String(options.column));
	args.push("--to", options.to);
	if (options.apply) args.push("--apply");
	if (options.workspace) args.push("--workspace", options.workspace);
	if (options.language) args.push("--language", options.language);
	if (options.cached) args.push("--cached");
	if (options.others) args.push("--others");
	if (options.ignored) args.push("--ignored");
	return parseRenameOutput(await runReadSeek(args, { signal: options.signal }));
}

// --- Identify ---

export interface IdentifierOutput {
	text: string;
	start_column: number;
	end_column: number;
	start_byte: number;
	end_byte: number;
}

export interface IdentifyOutput {
	file: string;
	language: string;
	engine?: string;
	line_count: number;
	file_hash: string;
	line: number;
	column: number;
	line_hash: string;
	identifier?: IdentifierOutput;
	symbol?: {
		name: string;
		kind: string;
		qualified_name: string;
		start_line: number;
		end_line: number;
	};
}

interface IdentifyOptions {
	line?: number;
	column?: number;
	language?: string;
	signal?: AbortSignal;
}

function parseIdentifyOutput(value: unknown): IdentifyOutput {
	if (!value || typeof value !== "object") throw new Error("invalid readseek identify output");
	const output = value as Record<string, unknown>;
	const identifier = output.identifier;
	const symbol = output.symbol;
	return {
		file: requireString(output.file, "file"),
		language: requireString(output.language, "language"),
		engine: optionalString(output.engine, "engine"),
		line_count: requireNumber(output.line_count, "line_count"),
		file_hash: requireString(output.file_hash, "file_hash"),
		line: requireNumber(output.line, "line"),
		column: requireNumber(output.column, "column"),
		line_hash: requireString(output.line_hash, "line_hash"),
		identifier: identifier && typeof identifier === "object"
			? {
				text: requireString((identifier as Record<string, unknown>).text, "identifier.text"),
				start_column: requireNumber((identifier as Record<string, unknown>).start_column, "identifier.start_column"),
				end_column: requireNumber((identifier as Record<string, unknown>).end_column, "identifier.end_column"),
				start_byte: requireNumber((identifier as Record<string, unknown>).start_byte, "identifier.start_byte"),
				end_byte: requireNumber((identifier as Record<string, unknown>).end_byte, "identifier.end_byte"),
			}
			: undefined,
		symbol: symbol && typeof symbol === "object"
			? {
				name: requireString((symbol as Record<string, unknown>).name, "symbol.name"),
				kind: requireString((symbol as Record<string, unknown>).kind, "symbol.kind"),
				qualified_name: requireString((symbol as Record<string, unknown>).qualified_name, "symbol.qualified_name"),
				start_line: requireNumber((symbol as Record<string, unknown>).start_line, "symbol.start_line"),
				end_line: requireNumber((symbol as Record<string, unknown>).end_line, "symbol.end_line"),
			}
			: undefined,
	};
}

export async function readSeekIdentify(
	filePath: string,
	content: string,
	options: IdentifyOptions = {},
): Promise<IdentifyOutput> {
	const target = options.line === undefined ? `stdin:${filePath}` : `stdin:${filePath}:${options.line}`;
	const args = ["identify", target];
	if (options.column !== undefined) args.push("--column", String(options.column));
	if (options.language) args.push("--language", options.language);
	return parseIdentifyOutput(await runReadSeek(args, { signal: options.signal, stdin: content }));
}

// --- Def ---

export interface DefLocation {
	file: string;
	line: number;
	column: number;
	line_hash: string;
	text: string;
	kind?: string;
	name?: string;
	qualified_name?: string;
}

interface DefOptions {
	name: string;
	language?: string;
	cached?: boolean;
	others?: boolean;
	ignored?: boolean;
	signal?: AbortSignal;
}

function parseDefCompact(value: unknown): DefLocation[] {
	if (!value || typeof value !== "object") throw new Error("invalid readseek def output");
	const output = value as Record<string, unknown>;
	const locations = output.locations;
	if (!Array.isArray(locations)) throw new Error("invalid readseek def locations");
	return locations.map((loc) => {
		if (!loc || typeof loc !== "object") throw new Error("invalid readseek def location");
		const item = loc as Record<string, unknown>;
		return {
			file: requireString(item.file, "location.file"),
			line: requireNumber(item.line, "location.line"),
			column: requireNumber(item.column, "location.column"),
			line_hash: requireString(item.line_hash, "location.line_hash"),
			text: requireString(item.text, "location.text"),
			kind: optionalString(item.kind, "location.kind"),
			name: optionalString(item.name, "location.name"),
			qualified_name: optionalString(item.qualified_name, "location.qualified_name"),
		};
	});
}

export async function readSeekDef(
	target: string,
	options: DefOptions,
): Promise<DefLocation[]> {
	const args = ["def", target, "--format", "plain", options.name];
	if (options.language) args.push("--language", options.language);
	if (options.cached) args.push("--cached");
	if (options.others) args.push("--others");
	if (options.ignored) args.push("--ignored");
	return parseDefCompact(await runReadSeek(args, { signal: options.signal }));
}
