import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Node environment: this project talks to Pinecone/OpenAI and uses
    // danfojs-node + onnxruntime, none of which belong in a browser env.
    environment: "node",
    include: ["tests/**/*.test.ts"],
    // The live e2e waits on Pinecone index creation + eventual consistency,
    // so it needs a generous ceiling. Unit tests finish well under this.
    testTimeout: 120_000,
    hookTimeout: 120_000,
  },
});
