import { afterEach, describe, expect, it } from "vitest";
import { sliceIntoChunks, getEnv } from "../../src/utils/util.js";

describe("sliceIntoChunks", () => {
  it("splits an array evenly when the length is a multiple of the chunk size", () => {
    expect(sliceIntoChunks([1, 2, 3, 4], 2)).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });

  it("leaves a smaller final chunk when the length is not a multiple", () => {
    expect(sliceIntoChunks([1, 2, 3, 4, 5], 2)).toEqual([[1, 2], [3, 4], [5]]);
  });

  it("returns a single chunk when the chunk size exceeds the array length", () => {
    expect(sliceIntoChunks([1, 2], 10)).toEqual([[1, 2]]);
  });

  it("returns no chunks for an empty array", () => {
    expect(sliceIntoChunks([], 3)).toEqual([]);
  });
});

describe("getEnv", () => {
  const KEY = "UTIL_TEST_ENV_VAR";

  afterEach(() => {
    delete process.env[KEY];
  });

  it("returns the value when the variable is set", () => {
    process.env[KEY] = "hello";
    expect(getEnv(KEY)).toBe("hello");
  });

  it("throws a descriptive error when the variable is missing", () => {
    expect(() => getEnv(KEY)).toThrowError(
      `${KEY} environment variable not set`
    );
  });

  it("throws when the variable is set to an empty string", () => {
    process.env[KEY] = "";
    expect(() => getEnv(KEY)).toThrowError(
      `${KEY} environment variable not set`
    );
  });
});
