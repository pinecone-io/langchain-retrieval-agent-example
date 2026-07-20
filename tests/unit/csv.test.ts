import { describe, it, expect } from "vitest";
import { parseCSV } from "../../src/utils/csv.js";

describe("parseCSV", () => {
  it("parses a simple CSV into row objects keyed by the header", () => {
    const csv = "title,author\nHello,Ada\nWorld,Alan";
    expect(parseCSV(csv)).toEqual([
      { title: "Hello", author: "Ada" },
      { title: "World", author: "Alan" },
    ]);
  });

  it("handles commas inside quoted fields", () => {
    const csv = 'title,article\n"A, B, C","one, two, three"';
    expect(parseCSV(csv)).toEqual([
      { title: "A, B, C", article: "one, two, three" },
    ]);
  });

  it("handles newlines inside quoted fields", () => {
    const csv = 'title,article\n"Multi","line one\nline two"';
    expect(parseCSV(csv)).toEqual([
      { title: "Multi", article: "line one\nline two" },
    ]);
  });

  it("unescapes doubled quotes inside quoted fields", () => {
    const csv = 'title,article\n"Quote","she said ""hi"" loudly"';
    expect(parseCSV(csv)).toEqual([
      { title: "Quote", article: 'she said "hi" loudly' },
    ]);
  });

  it("handles CRLF line endings", () => {
    const csv = "title,author\r\nHello,Ada\r\nWorld,Alan\r\n";
    expect(parseCSV(csv)).toEqual([
      { title: "Hello", author: "Ada" },
      { title: "World", author: "Alan" },
    ]);
  });

  it("ignores blank lines and a trailing newline", () => {
    const csv = "title,author\nHello,Ada\n\nWorld,Alan\n";
    expect(parseCSV(csv)).toEqual([
      { title: "Hello", author: "Ada" },
      { title: "World", author: "Alan" },
    ]);
  });

  it("fills missing trailing columns with empty strings", () => {
    const csv = "title,author,section\nHello,Ada";
    expect(parseCSV(csv)).toEqual([
      { title: "Hello", author: "Ada", section: "" },
    ]);
  });

  it("returns an empty array for empty input", () => {
    expect(parseCSV("")).toEqual([]);
  });
});
