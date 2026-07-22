import { describe, expect, it } from "vitest";
import { dropDuplicatesBy } from "../../src/utils/dataLoader.js";

describe("dropDuplicatesBy", () => {
  it("keeps only the first row seen for each distinct value in the target column", () => {
    const rows = [
      { context: "alpha", id: "1" },
      { context: "alpha", id: "2" },
      { context: "beta", id: "3" },
      { context: "gamma", id: "4" },
    ];

    const result = dropDuplicatesBy(rows, "context");

    expect(result).toEqual([
      { context: "alpha", id: "1" },
      { context: "beta", id: "3" },
      { context: "gamma", id: "4" },
    ]);
  });

  it("is a no-op when the target column has no duplicates", () => {
    const rows = [
      { context: "one", id: "1" },
      { context: "two", id: "2" },
      { context: "three", id: "3" },
    ];

    const result = dropDuplicatesBy(rows, "context");

    expect(result).toEqual(rows);
  });
});
