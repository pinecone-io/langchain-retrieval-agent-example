import { describe, expect, it } from "vitest";
import * as dfd from "danfojs-node";
import { dropDuplicates } from "../../src/utils/dataLoader.js";

describe("dropDuplicates", () => {
  it("removes rows whose values in the target column are duplicated", () => {
    const df = new dfd.DataFrame({
      context: ["alpha", "alpha", "beta", "gamma"],
      id: ["1", "2", "3", "4"],
    });

    const result = dropDuplicates(df, "context");

    // "alpha" appears twice; only its first occurrence survives.
    expect(result.shape[0]).toBe(3);
    expect(result["context"].values).toEqual(["alpha", "beta", "gamma"]);
  });

  it("is a no-op (by row count) when the target column has no duplicates", () => {
    const df = new dfd.DataFrame({
      context: ["one", "two", "three"],
      id: ["1", "2", "3"],
    });

    const result = dropDuplicates(df, "context");

    expect(result.shape[0]).toBe(3);
    expect(result["context"].values).toEqual(["one", "two", "three"]);
  });
});
