import { describe, expect, it, vi } from "vitest";

const { getLanguageFromPathMock, highlightCodeMock } = vi.hoisted(() => ({
	getLanguageFromPathMock: vi.fn((path: string) => path.endsWith(".ts") ? "typescript" : undefined),
	highlightCodeMock: vi.fn((code: string, language?: string) =>
		code.split("\n").map((line) => `<${language ?? "plain"}>${line}</${language ?? "plain"}>`),
	),
}));

vi.mock("@earendil-works/pi-coding-agent", async () => ({
	...(await import("./support/pi-coding-agent-mock.js")).createPiCodingAgentBaseMock(),
	getLanguageFromPath: getLanguageFromPathMock,
	highlightCode: highlightCodeMock,
}));

const { renderGrepSourceForDisplay, renderReadSourceForDisplay, renderReadSourceForDisplayCached } = await import("../src/tui-source-render.js");

describe("renderReadSourceForDisplay", () => {
	it("removes known anchors and highlights source blocks", () => {
		const input = "warning\n\n1:abc|const answer = 42;\n2:def|return answer;\n3:999|unchanged";
		const rendered = renderReadSourceForDisplay(input, "example.ts", new Set(["1:abc", "2:def"]), undefined);

		expect(highlightCodeMock).toHaveBeenCalledWith("const answer = 42;\nreturn answer;", "typescript");
		expect(rendered).toContain("<typescript>const answer = 42;</typescript>");
		expect(rendered).toContain("<typescript>return answer;</typescript>");
		expect(rendered).not.toContain("1:abc|");
		expect(rendered).not.toContain("2:def|");
		expect(rendered).toContain("3:999|unchanged");
	});

	it("caches highlighted output until rendering inputs change", () => {
		highlightCodeMock.mockClear();
		const key = {};
		const anchors = new Set(["1:abc"]);
		const theme = { fg: (style: string, text: string) => `<${style}>${text}` };

		renderReadSourceForDisplayCached(key, "1:abc|const value = 1;", "example.ts", anchors, 80, theme);
		renderReadSourceForDisplayCached(key, "1:abc|const value = 1;", "example.ts", anchors, 80, theme);

		expect(highlightCodeMock).toHaveBeenCalledTimes(1);

		const changedTheme = { fg: (style: string, text: string) => `<changed-${style}>${text}` };
		renderReadSourceForDisplayCached(key, "1:abc|const value = 1;", "example.ts", anchors, 80, changedTheme);
		expect(highlightCodeMock).toHaveBeenCalledTimes(2);
	});
});

describe("renderGrepSourceForDisplay", () => {
	it("converts known match and context anchors to grep lines", () => {
		const input = [
			"[1 matches in 1 files]",
			"--- src/example.ts (1 matches) ---",
			"src/example.ts:  6:def|before();",
			"src/example.ts:>>7:abc|target();",
			"src/example.ts:>>8:999|unchanged();",
		].join("\n");
		const rendered = renderGrepSourceForDisplay(input, new Set(["6:def", "7:abc"]));

		expect(rendered).toContain("src/example.ts-6-before();");
		expect(rendered).toContain("src/example.ts:7:target();");
		expect(rendered).not.toContain("6:def|");
		expect(rendered).not.toContain("7:abc|");
		expect(rendered).toContain("src/example.ts:>>8:999|unchanged();");
	});

	it("handles Windows drive-letter paths", () => {
		const input = "C:\\src\\example.ts:>>7:abc|target();";
		const rendered = renderGrepSourceForDisplay(input, new Set(["7:abc"]));

		expect(rendered).toBe("C:\\src\\example.ts:7:target();");
	});
});
