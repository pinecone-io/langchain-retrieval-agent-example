
const sliceIntoChunks = <T>(arr: T[], chunkSize: number) => Array.from({ length: Math.ceil(arr.length / chunkSize) }, (_, i) =>
  arr.slice(i * chunkSize, (i + 1) * chunkSize)
);

export const getEnv = (key: string): string => {
  const value = process.env[key];
  if (!value) {
    throw new Error(`${key} environment variable not set`);
  }
  return value;
};

const validateEnvironmentVariables = () => {
  getEnv("PINECONE_API_KEY");
  getEnv("PINECONE_ENVIRONMENT");
  getEnv("PINECONE_INDEX");
};

export {
  sliceIntoChunks,
  validateEnvironmentVariables
};
