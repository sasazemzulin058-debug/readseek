import { readFileSync, statSync, type Stats } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export type ReadSeekImageAnalysisMode = "on" | "off" | "auto";
export type ReadSeekSyntaxValidationMode = "warn" | "block" | "off";
export type ReadSeekDisplayMode = "compact" | "expanded";
export interface ReadSeekDisplaySettings {
  read?: ReadSeekDisplayMode;
  grep?: ReadSeekDisplayMode;
  edit?: ReadSeekDisplayMode;
  write?: ReadSeekDisplayMode;
}
interface ReadSeekGrepSettings {
  maxLines?: number;
  maxBytes?: number;
}

const READSEEK_TOOL_REPLACEMENTS = {
  read: "readSeek_read",
  edit: "readSeek_edit",
  grep: "readSeek_grep",
  write: "readSeek_write",
} as const;

type ReadSeekReplacementTool = keyof typeof READSEEK_TOOL_REPLACEMENTS;

interface ReadSeekJsonSettings {
  replacedTools?: ReadSeekReplacementTool[];
  imageMode?: ReadSeekImageAnalysisMode;
  syntaxValidation?: ReadSeekSyntaxValidationMode;
  timeoutMs?: number;
  grep?: ReadSeekGrepSettings;
  display?: ReadSeekDisplaySettings;
}

export interface ReadSeekSettingsWarning {
  source: string;
  message: string;
  path?: string;
}

export interface ReadSeekSettingsResult {
  settings: ReadSeekJsonSettings;
  warnings: ReadSeekSettingsWarning[];
}

const READSEEK_KEYS = ["replacedTools", "imageMode", "syntaxValidation", "timeoutMs", "grep", "display"];
const READSEEK_GREP_KEYS = ["maxLines", "maxBytes"];
const READSEEK_DISPLAY_KEYS = ["read", "grep", "edit", "write"];

const DEFAULT_DISPLAY_MODES: Record<ReadSeekReplacementTool, ReadSeekDisplayMode> = {
  read: "compact",
  grep: "compact",
  edit: "expanded",
  write: "expanded",
};

function globalSettingsPath(): string {
  return join(homedir(), ".pi/agent/settings.json");
}

function projectSettingsPath(): string {
  return join(process.cwd(), ".pi/settings.json");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function invalid(source: string, path: string): ReadSeekSettingsWarning {
  return { source, path, message: `Invalid readseek setting at ${path}` };
}

function warnUnknownKeys(
  raw: Record<string, unknown>,
  known: string[],
  prefix: string,
  source: string,
  warnings: ReadSeekSettingsWarning[],
): void {
  for (const key of Object.keys(raw)) {
    if (known.includes(key)) continue;
    const path = `${prefix}.${key}`;
    warnings.push({ source, path, message: `Unknown readseek setting at ${path}` });
  }
}

function readPositive(
  raw: Record<string, unknown>,
  key: string,
  path: string,
  source: string,
  warnings: ReadSeekSettingsWarning[],
): number | undefined {
  if (!(key in raw)) return undefined;
  const val = raw[key];
  if (typeof val === "number" && Number.isSafeInteger(val) && val > 0) return val;
  warnings.push(invalid(source, path));
  return undefined;
}

function readEnum<T extends string>(
  raw: Record<string, unknown>,
  key: string,
  values: readonly T[],
  path: string,
  source: string,
  warnings: ReadSeekSettingsWarning[],
): T | undefined {
  if (!(key in raw)) return undefined;
  const val = raw[key];
  if (typeof val === "string" && (values as readonly string[]).includes(val)) return val as T;
  warnings.push(invalid(source, path));
  return undefined;
}

function readImageMode(
  raw: Record<string, unknown>,
  source: string,
  warnings: ReadSeekSettingsWarning[],
): ReadSeekImageAnalysisMode | undefined {
  if (!("imageMode" in raw)) return undefined;
  const val = raw.imageMode;
  if (val === "on" || val === "off" || val === "auto") return val;
  warnings.push(invalid(source, "readseek.imageMode"));
  return undefined;
}

function readReplacedTools(
  raw: Record<string, unknown>,
  source: string,
  warnings: ReadSeekSettingsWarning[],
): ReadSeekReplacementTool[] | undefined {
  if (!("replacedTools" in raw)) return undefined;
  const value = raw.replacedTools;
  if (!Array.isArray(value)) {
    warnings.push(invalid(source, "readseek.replacedTools"));
    return undefined;
  }

  const tools: ReadSeekReplacementTool[] = [];
  value.forEach((tool, index) => {
    if (typeof tool === "string" && Object.hasOwn(READSEEK_TOOL_REPLACEMENTS, tool)) {
      tools.push(tool as ReadSeekReplacementTool);
    } else {
      warnings.push(invalid(source, `readseek.replacedTools[${index}]`));
    }
  });
  return tools;
}
function readDisplaySettings(
  raw: Record<string, unknown>,
  source: string,
  warnings: ReadSeekSettingsWarning[],
): ReadSeekDisplaySettings | undefined {
  if (!("display" in raw)) return undefined;
  if (!isRecord(raw.display)) {
    warnings.push(invalid(source, "readseek.display"));
    return undefined;
  }
  warnUnknownKeys(raw.display, READSEEK_DISPLAY_KEYS, "readseek.display", source, warnings);
  const display: ReadSeekDisplaySettings = {};
  for (const tool of READSEEK_DISPLAY_KEYS as Array<keyof ReadSeekDisplaySettings>) {
    const mode = readEnum(
      raw.display,
      tool,
      ["compact", "expanded"] as const,
      `readseek.display.${tool}`,
      source,
      warnings,
    );
    if (mode !== undefined) display[tool] = mode;
  }
  return Object.keys(display).length > 0 ? display : undefined;
}

function validateSettings(raw: unknown, source: string): ReadSeekSettingsResult {
  const settings: ReadSeekJsonSettings = {};
  const warnings: ReadSeekSettingsWarning[] = [];
  if (!isRecord(raw)) {
    warnings.push({ source, message: "Invalid readseek settings: expected a JSON object" });
    return { settings, warnings };
  }
  if (!("readseek" in raw)) {
    const misplaced = READSEEK_KEYS.filter((key) => key in raw);
    if (misplaced.length > 0) {
      warnings.push({
        source,
        path: misplaced[0],
        message: `Readseek setting at top level: move ${misplaced.join(", ")} under "readseek"`,
      });
    }
    return { settings, warnings };
  }
  if (!isRecord(raw.readseek)) {
    warnings.push(invalid(source, "readseek"));
    return { settings, warnings };
  }

  const section = raw.readseek;
  warnUnknownKeys(section, READSEEK_KEYS, "readseek", source, warnings);

  const replacedTools = readReplacedTools(section, source, warnings);
  if (replacedTools !== undefined) settings.replacedTools = replacedTools;

  const imageMode = readImageMode(section, source, warnings);
  if (imageMode !== undefined) settings.imageMode = imageMode;

  const syntaxValidation = readEnum(
    section,
    "syntaxValidation",
    ["warn", "block", "off"] as const,
    "readseek.syntaxValidation",
    source,
    warnings,
  );
  if (syntaxValidation !== undefined) settings.syntaxValidation = syntaxValidation;

  const timeoutMs = readPositive(section, "timeoutMs", "readseek.timeoutMs", source, warnings);
  if (timeoutMs !== undefined) settings.timeoutMs = timeoutMs;

  if ("grep" in section) {
    if (isRecord(section.grep)) {
      warnUnknownKeys(section.grep, READSEEK_GREP_KEYS, "readseek.grep", source, warnings);
      const grep: ReadSeekGrepSettings = {};
      const maxLines = readPositive(section.grep, "maxLines", "readseek.grep.maxLines", source, warnings);
      if (maxLines !== undefined) grep.maxLines = maxLines;
      const maxBytes = readPositive(section.grep, "maxBytes", "readseek.grep.maxBytes", source, warnings);
      if (maxBytes !== undefined) grep.maxBytes = maxBytes;
      if (Object.keys(grep).length > 0) settings.grep = grep;
    } else {
      warnings.push(invalid(source, "readseek.grep"));
    }
  }
  const display = readDisplaySettings(section, source, warnings);
  if (display !== undefined) settings.display = display;

  return { settings, warnings };
}

interface SettingsFileCacheEntry {
  mtimeMs: number;
  size: number;
  result: ReadSeekSettingsResult;
}

const settingsFileCache = new Map<string, SettingsFileCacheEntry>();

function statFileOrNull(path: string): Stats | null {
  try {
    return statSync(path);
  } catch {
    return null;
  }
}

function readSettingsFile(path: string): ReadSeekSettingsResult {
  const stats = statFileOrNull(path);
  if (!stats) {
    settingsFileCache.delete(path);
    return { settings: {}, warnings: [] };
  }

  const cached = settingsFileCache.get(path);
  if (cached && cached.mtimeMs === stats.mtimeMs && cached.size === stats.size) return cached.result;

  let result: ReadSeekSettingsResult;
  try {
    const text = readFileSync(path, "utf8");
    result = validateSettings(JSON.parse(text) as unknown, path);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    result = { settings: {}, warnings: [{ source: path, message: `Invalid JSON: ${message}` }] };
  }
  settingsFileCache.set(path, { mtimeMs: stats.mtimeMs, size: stats.size, result });
  return result;
}

function mergeSettings(base: ReadSeekJsonSettings, override: ReadSeekJsonSettings): ReadSeekJsonSettings {
  const settings: ReadSeekJsonSettings = { ...base, ...override };
  const grep = { ...(base.grep ?? {}), ...(override.grep ?? {}) };
  if (Object.keys(grep).length > 0) settings.grep = grep;
  const display = { ...(base.display ?? {}), ...(override.display ?? {}) };
  if (Object.keys(display).length > 0) settings.display = display;
  return settings;
}

export function resolveReadSeekJsonSettings(): ReadSeekSettingsResult {
  const globalResult = readSettingsFile(globalSettingsPath());
  const projectResult = readSettingsFile(projectSettingsPath());
  return {
    settings: mergeSettings(globalResult.settings, projectResult.settings),
    warnings: [...globalResult.warnings, ...projectResult.warnings],
  };
}

export function resolveReadSeekImageMode(): ReadSeekImageAnalysisMode {
  return resolveReadSeekJsonSettings().settings.imageMode ?? "auto";
}

export function resolveReadSeekSyntaxValidation(): ReadSeekSyntaxValidationMode | undefined {
  return resolveReadSeekJsonSettings().settings.syntaxValidation;
}

export function resolveReadSeekTimeoutMs(): number | undefined {
  return resolveReadSeekJsonSettings().settings.timeoutMs;
}
export function resolveReadSeekToolDisplayMode(tool: ReadSeekReplacementTool): ReadSeekDisplayMode {
  return resolveReadSeekJsonSettings().settings.display?.[tool] ?? DEFAULT_DISPLAY_MODES[tool];
}
