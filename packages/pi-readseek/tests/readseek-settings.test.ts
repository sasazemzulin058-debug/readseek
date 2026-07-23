import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { homeDir } = vi.hoisted(() => ({
	homeDir: { value: "" },
}));

vi.mock("node:os", async (importOriginal) => {
	const actual = await importOriginal<typeof import("node:os")>();
	return {
		...actual,
		homedir: () => homeDir.value,
	};
});

const {
	resolveReadSeekJsonSettings,
	resolveReadSeekImageMode,
	resolveReadSeekSyntaxValidation,
	resolveReadSeekToolDisplayMode,
} = await import("../src/readseek-settings.js");

describe("readseek settings", () => {
	let tempHome: string;
	let tempCwd: string;
	let previousCwd: string;

	beforeEach(async () => {
		previousCwd = process.cwd();
		tempHome = await mkdtemp(path.join(tmpdir(), "pi-readseek-home-"));
		tempCwd = await mkdtemp(path.join(tmpdir(), "pi-readseek-cwd-"));
		homeDir.value = tempHome;
		process.chdir(tempCwd);
	});

	afterEach(async () => {
		process.chdir(previousCwd);
		await rm(tempHome, { recursive: true, force: true });
		await rm(tempCwd, { recursive: true, force: true });
	});

	async function writeGlobal(settings: unknown) {
		await mkdir(path.join(tempHome, ".pi", "agent"), { recursive: true });
		await writeFile(path.join(tempHome, ".pi", "agent", "settings.json"), JSON.stringify(settings));
	}

	async function writeProject(settings: unknown) {
		await mkdir(path.join(tempCwd, ".pi"), { recursive: true });
		await writeFile(path.join(tempCwd, ".pi", "settings.json"), JSON.stringify(settings));
	}

	it("defaults imageMode to auto", () => {
		expect(resolveReadSeekImageMode()).toBe("auto");
	});

	it("reads imageMode from global settings", async () => {
		await writeGlobal({ readseek: { imageMode: "auto" } });
		expect(resolveReadSeekImageMode()).toBe("auto");
	});

	it("accepts on as an imageMode", async () => {
		await writeGlobal({ readseek: { imageMode: "on" } });
		expect(resolveReadSeekImageMode()).toBe("on");
	});

	it("lets project settings override global", async () => {
		await writeGlobal({ readseek: { imageMode: "auto" } });
		await writeProject({ readseek: { imageMode: "off" } });
		expect(resolveReadSeekImageMode()).toBe("off");
	});

	it("warns on invalid imageMode and falls back to auto", async () => {
		await writeGlobal({ readseek: { imageMode: "maybe" } });
		const { settings, warnings } = resolveReadSeekJsonSettings();
		expect(settings.imageMode).toBeUndefined();
		expect(warnings).toHaveLength(1);
		expect(warnings[0]?.path).toBe("readseek.imageMode");
		expect(resolveReadSeekImageMode()).toBe("auto");
	});

	it("merges nested grep settings", async () => {
		await writeGlobal({ readseek: { grep: { maxLines: 50, maxBytes: 1000 } } });
		await writeProject({ readseek: { grep: { maxLines: 25 } } });
		expect(resolveReadSeekJsonSettings().settings.grep).toEqual({ maxLines: 25, maxBytes: 1000 });
	});

	it("reads replacedTools and syntaxValidation", async () => {
		await writeGlobal({ readseek: { replacedTools: ["read", "edit"], syntaxValidation: "block" } });
		expect(resolveReadSeekJsonSettings().settings.replacedTools).toEqual(["read", "edit"]);
		expect(resolveReadSeekSyntaxValidation()).toBe("block");
	});

	it("keeps valid replacedTools entries and warns about unsupported tools", async () => {
		await writeGlobal({ readseek: { replacedTools: ["read", "readSeek_read", "bash"] } });
		const { settings, warnings } = resolveReadSeekJsonSettings();
		expect(warnings.map((warning) => warning.path)).toEqual(["readseek.replacedTools[1]", "readseek.replacedTools[2]"]);
		expect(settings.replacedTools).toEqual(["read"]);
	});

	it("warns on unknown keys in the readseek section", async () => {
		await writeGlobal({ readseek: { imagemode: "off", grep: { maxlines: 10 } } });
		const { warnings } = resolveReadSeekJsonSettings();
		expect(warnings.map((warning) => warning.path)).toEqual(["readseek.imagemode", "readseek.grep.maxlines"]);
	});

	it("warns when a readseek setting is at the top level", async () => {
		await writeGlobal({ imageMode: "off", theme: "blackboard-pro" });
		const { settings, warnings } = resolveReadSeekJsonSettings();
		expect(warnings).toHaveLength(1);
		expect(warnings[0]?.path).toBe("imageMode");
		expect(settings.imageMode).toBeUndefined();
	});

	it("does not warn when a non-readseek settings file omits the readseek section", async () => {
		await writeGlobal({ packages: ["npm:pi-readseek"], theme: "blackboard-pro" });
		const { settings, warnings } = resolveReadSeekJsonSettings();
		expect(warnings).toEqual([]);
		expect(settings).toEqual({});
	});

	it("picks up settings changes and deletions despite caching", async () => {
		await writeGlobal({ readseek: { imageMode: "auto" } });
		expect(resolveReadSeekImageMode()).toBe("auto");
		expect(resolveReadSeekImageMode()).toBe("auto");

		await writeGlobal({ readseek: { imageMode: "off" } });
		expect(resolveReadSeekImageMode()).toBe("off");

		await rm(path.join(tempHome, ".pi", "agent", "settings.json"));
		expect(resolveReadSeekImageMode()).toBe("auto");
	});

	it("provides default display modes per tool", () => {
		expect(resolveReadSeekToolDisplayMode("read")).toBe("compact");
		expect(resolveReadSeekToolDisplayMode("grep")).toBe("compact");
		expect(resolveReadSeekToolDisplayMode("edit")).toBe("expanded");
		expect(resolveReadSeekToolDisplayMode("write")).toBe("expanded");
	});

	it("parses display mode settings and allows overrides", async () => {
		await writeGlobal({ readseek: { display: { read: "expanded", edit: "compact" } } });
		expect(resolveReadSeekToolDisplayMode("read")).toBe("expanded");
		expect(resolveReadSeekToolDisplayMode("grep")).toBe("compact");
		expect(resolveReadSeekToolDisplayMode("edit")).toBe("compact");
		expect(resolveReadSeekToolDisplayMode("write")).toBe("expanded");
	});

	it("merges display mode settings from project overrides", async () => {
		await writeGlobal({ readseek: { display: { read: "expanded", grep: "expanded" } } });
		await writeProject({ readseek: { display: { read: "compact" } } });
		expect(resolveReadSeekToolDisplayMode("read")).toBe("compact");
		expect(resolveReadSeekToolDisplayMode("grep")).toBe("expanded");
	});

	it("warns on unknown and invalid display mode settings", async () => {
		await writeGlobal({ readseek: { display: { read: "invalid", unknownTool: "compact" } } });
		const { warnings } = resolveReadSeekJsonSettings();
		expect(warnings.map((w) => w.path)).toEqual([
			"readseek.display.unknownTool",
			"readseek.display.read",
		]);
		expect(resolveReadSeekToolDisplayMode("read")).toBe("compact");
	});
});
