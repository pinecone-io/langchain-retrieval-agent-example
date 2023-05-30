import { dataFrameFromURL } from "./dataLoader.js";

const url =
  "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json";

const loadSquad = async () => {
  const df = await dataFrameFromURL(
    url,
    [
      "context",
      "qas.id",
      "qas.question",
      "qas.answers.answer_start",
      "qas.answers.text",
    ],
    ["qas", "qas.answers"]
  );

  return df;
};

export { loadSquad };