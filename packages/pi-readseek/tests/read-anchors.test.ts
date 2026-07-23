import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

import { beforeEach, describe, expect, it, vi } from "vitest";

const { readSeekMapMock, readSeekReadMock, readSeekDetectMock, readSeekImageMock, readSeekPdfMock, readSeekPreparedImageMock, imageMode } = vi.hoisted(() => ({
	readSeekMapMock: vi.fn(),
	readSeekReadMock: vi.fn(),
	readSeekDetectMock: vi.fn(),
	readSeekImageMock: vi.fn(),
	readSeekPdfMock: vi.fn(),
	readSeekPreparedImageMock: vi.fn(),
	imageMode: { value: "auto" as "on" | "off" | "auto" },
}));

vi.mock("@earendil-works/pi-coding-agent", async () => ({
	...(await import("./support/pi-coding-agent-mock.js")).createPiCodingAgentBaseMock(),
}));

vi.mock("../src/readseek-settings.js", () => ({
	resolveReadSeekImageMode: () => imageMode.value,
	resolveReadSeekJsonSettings: () => ({ settings: {}, warnings: [] }),
	resolveReadSeekSyntaxValidation: () => undefined,
	resolveReadSeekTimeoutMs: () => undefined,
}));

vi.mock("../src/readseek-client.js", () => ({
	readSeekMap: readSeekMapMock,
	readSeekRead: readSeekReadMock,
	readSeekDetect: readSeekDetectMock,
	readSeekImage: readSeekImageMock,
	readSeekPdf: readSeekPdfMock,
	readSeekPreparedImage: readSeekPreparedImageMock,
}));

const { executeRead } = await import("../src/read.js");

describe("executeRead anchor tracking", () => {
	beforeEach(() => {
		vi.clearAllMocks();
		imageMode.value = "auto";
	});

	async function writeImage(cwd: string): Promise<string> {
		const filePath = path.join(cwd, "image.png");
		const png = Buffer.from(
			"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/pLvAAAAAElFTkSuQmCC",
			"base64",
		);
		await writeFile(filePath, png);
		return filePath;
	}

	function imageDetectionFor(filePath: string) {
		return {
			kind: "image",
			type: "image/png",
			file: filePath,
			mime: "image/png",
			format: "png",
			width: 1,
			height: 1,
			animated: false,
		};
	}

	function mockImageDetection(filePath: string) {
		const imageDetection = imageDetectionFor(filePath);
		readSeekDetectMock.mockResolvedValue(imageDetection);
		readSeekImageMock.mockImplementation((_filePath: string, modes: string[]) =>
			Promise.resolve({
				...imageDetection,
				...(modes.includes("ocr") ? { ocr: "OCR TEXT" } : {}),
				...(modes.includes("caption") ? { caption: "A tiny test image." } : {}),
				...(modes.includes("objects") ? { objects: [{ label: "dot", bbox: [1, 2, 3, 4] }] } : {}),
			}),
		);
	}

	it("marks text reads with readseek lines as anchored", async () => {
		const cwd = await mkdtemp(path.join(tmpdir(), "pi-readseek-read-"));
		try {
			const filePath = path.join(cwd, "file.txt");
			await writeFile(filePath, "hello\nworld\n", "utf8");
			readSeekReadMock.mockResolvedValueOnce({
				file: filePath,
				language: "Text",
				line_count: 2,
				file_hash: "filehash",
				start_line: 1,
				end_line: 2,
				hashlines: [
					{ line: 1, hash: "aaa", text: "hello" },
					{ line: 2, hash: "bbb", text: "world" },
				],
			});
			const onSuccessfulRead = vi.fn();

			const result = await executeRead({
				toolCallId: "test",
				params: { path: "file.txt" },
				signal: undefined,
				onUpdate: undefined,
				cwd,
				onSuccessfulRead,
			});

			expect(result.content[0]).toMatchObject({
				type: "text",
				text: expect.stringContaining("1:aaa|hello\n2:bbb|world"),
			});

			expect(onSuccessfulRead).toHaveBeenCalledWith(filePath);
		} finally {
			await rm(cwd, { recursive: true, force: true });
		}
	});

	it("returns an error when requested image analysis fails", async () => {
		const cwd = await mkdtemp(path.join(tmpdir(), "pi-readseek-read-"));
		try {
			const filePath = await writeImage(cwd);
			readSeekDetectMock.mockResolvedValue(imageDetectionFor(filePath));
			readSeekImageMock.mockRejectedValueOnce(new Error("readseek crashed with SIGFPE"));
			const result = await executeRead({
				toolCallId: "test",
				params: { path: "image.png", image: "ocr" },
				signal: undefined,
				onUpdate: undefined,
				cwd,
			});

			expect((result as { isError?: boolean }).isError).toBe(true);
			expect((result.content[0] as { text: string }).text).toContain("Image analysis unavailable");
		} finally {
			await rm(cwd, { recursive: true, force: true });
		}
	});

	it("returns explicitly selected local image analysis", async () => {
		const cwd = await mkdtemp(path.join(tmpdir(), "pi-readseek-read-"));
		try {
			const filePath = await writeImage(cwd);
			mockImageDetection(filePath);
			const onSuccessfulRead = vi.fn();

			const result = await executeRead({
				toolCallId: "test",
				params: { path: "image.png", image: "caption" },
				signal: undefined,
				onUpdate: undefined,
				cwd,
				onSuccessfulRead,
			});

			expect(onSuccessfulRead).not.toHaveBeenCalled();
			const text = (result.content as Array<{ type: string; text: string }>).map((part) => part.text).join("\n");
			expect(text).toContain("Image caption:\nA tiny test image.");
			expect(readSeekImageMock).toHaveBeenCalledWith(filePath, ["caption"], { signal: undefined });
		} finally {
			await rm(cwd, { recursive: true, force: true });
		}
	});

	it("reports an empty image analysis as a valid result", async () => {
		const cwd = await mkdtemp(path.join(tmpdir(), "pi-readseek-read-"));
		try {
			const filePath = await writeImage(cwd);
			mockImageDetection(filePath);
			readSeekImageMock.mockResolvedValueOnce({
				...imageDetectionFor(filePath),
				objects: [],
			});

			const result = await executeRead({
				toolCallId: "test",
				params: { path: "image.png", image: "objects" },
				signal: undefined,
				onUpdate: undefined,
				cwd,
			});

			expect((result as { isError?: boolean }).isError).not.toBe(true);
			expect((result.content[0] as { text: string }).text).toBe("No objects detected in image.");
		} finally {
			await rm(cwd, { recursive: true, force: true });
		}
	});

	it("skips images when imageMode is off", async () => {
		const cwd = await mkdtemp(path.join(tmpdir(), "pi-readseek-read-"));
		try {
			imageMode.value = "off";
			const filePath = await writeImage(cwd);
			mockImageDetection(filePath);

			const result = await executeRead({
				toolCallId: "test",
				params: { path: "image.png" },
				signal: undefined,
				onUpdate: undefined,
				cwd,
			});

			expect((result.content[0] as { text: string }).text).toContain("Skipped image/PDF");
			expect(readSeekImageMock).not.toHaveBeenCalled();
		} finally {
			await rm(cwd, { recursive: true, force: true });
		}
	});

	it("rejects page selection for images", async () => {
		const cwd = await mkdtemp(path.join(tmpdir(), "pi-readseek-read-"));
		try {
			const filePath = await writeImage(cwd);
			mockImageDetection(filePath);

			const result = await executeRead({
				toolCallId: "test",
				params: { path: "image.png", image: "ocr", page: 2 },
				signal: undefined,
				onUpdate: undefined,
				cwd,
			});

			expect((result as { isError?: boolean }).isError).toBe(true);
			expect((result.content[0] as { text: string }).text).toContain("page parameter applies to PDFs only");
			expect(readSeekImageMock).not.toHaveBeenCalled();
		} finally {
			await rm(cwd, { recursive: true, force: true });
		}
	});

	it("preprocesses images when the model selects none in auto mode", async () => {
		const cwd = await mkdtemp(path.join(tmpdir(), "pi-readseek-read-"));
		try {
			imageMode.value = "auto";
			const filePath = await writeImage(cwd);
			mockImageDetection(filePath);
			readSeekPreparedImageMock.mockResolvedValueOnce({
				mime: "image/jpeg",
				encoding: "base64",
				data: "prepared-image",
			});

			const result = await executeRead({
				toolCallId: "test",
				params: { path: "image.png", image: "none" },
				signal: undefined,
				onUpdate: undefined,
				cwd,
			});

			expect(result.content).toEqual([{ type: "image", mimeType: "image/jpeg", data: "prepared-image" }]);
			expect(readSeekImageMock).not.toHaveBeenCalled();
		} finally {
			await rm(cwd, { recursive: true, force: true });
		}
	});

	it("returns PDF markdown and page-associated prepared images", async () => {
		const cwd = await mkdtemp(path.join(tmpdir(), "pi-readseek-read-"));
		try {
			const filePath = path.join(cwd, "paper.pdf");
			await writeFile(filePath, Buffer.from("%PDF-1.4\n"));
			readSeekDetectMock.mockResolvedValue({
				kind: "pdf",
				type: "application/pdf",
				file: filePath,
				mime: "application/pdf",
				format: "pdf",
				pages: 3,
			});
			readSeekPdfMock.mockResolvedValue({
				format: "pdf",
				pages: 3,
				markdown: "<!-- readseek:page 3 -->\nHello\n",
				images: [{ page: 3, width: 1, height: 1, mime: "image/png", mode: "none", encoding: "base64", data: "pixel" }],
			});

			const result = await executeRead({
				toolCallId: "test",
				params: { path: "paper.pdf", image: "none", page: 3 },
				signal: undefined,
				onUpdate: undefined,
				cwd,
			});

			expect(result.content).toEqual([
				{ type: "text", text: "<!-- readseek:page 3 -->\nHello\n" },
				{ type: "text", text: "[PDF page 3 image]" },
				{ type: "image", data: "pixel", mimeType: "image/png" },
			]);
			expect(readSeekPdfMock).toHaveBeenCalledWith(filePath, "none", { page: 3, signal: undefined });
		} finally {
			await rm(cwd, { recursive: true, force: true });
		}
	});

	it.each(["map", "local"])("treats %s bundle without symbol as a map read", async (bundle) => {
		const cwd = await mkdtemp(path.join(tmpdir(), "pi-readseek-read-"));
		try {
			const filePath = path.join(cwd, "file.ts");
			await writeFile(filePath, "const value = 1;\n", "utf8");
			readSeekReadMock.mockResolvedValueOnce({
				file: filePath,
				language: "TypeScript",
				line_count: 1,
				file_hash: "filehash",
				start_line: 1,
				end_line: 1,
				hashlines: [{ line: 1, hash: "aaa", text: "const value = 1;" }],
			});
			readSeekMapMock.mockResolvedValueOnce({
				path: filePath,
				totalLines: 1,
				totalBytes: 17,
				language: "TypeScript",
				symbols: [],
				detailLevel: "full",
			});

			const result = await executeRead({
				toolCallId: "test",
				params: { path: "file.ts", bundle },
				signal: undefined,
				onUpdate: undefined,
				cwd,
			});

			expect((result as { isError?: boolean }).isError).not.toBe(true);
			expect((result.details as any).readSeekValue.map).toEqual({
				requested: true,
				appended: true,
			});
		} finally {
			await rm(cwd, { recursive: true, force: true });
		}
	});
});
