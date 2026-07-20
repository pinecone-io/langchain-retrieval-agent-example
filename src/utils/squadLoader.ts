import { rowsFromURL, dropDuplicatesBy } from "./dataLoader.js";

const url =
  "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json";

interface SquadRecord {
  context: string;
  id: string;
  question: string;
  answer: string;
}

const loadSquad = async (): Promise<SquadRecord[]> => {
  const rows = await rowsFromURL(
    url,
    [
      "title",
      "paragraphs.context",
      "paragraphs.qas.id",
      "paragraphs.qas.question",
      "paragraphs.qas.answers.text",
    ],
    ["paragraphs", "paragraphs.qas", "paragraphs.qas.answers"]
  );

  const records: SquadRecord[] = rows.map((row) => ({
    context: row["paragraphs.context"],
    id: row["paragraphs.qas.id"],
    question: row["paragraphs.qas.question"],
    answer: row["paragraphs.qas.answers.text"],
  }));

  return dropDuplicatesBy(records, "context");
};

export { loadSquad, SquadRecord };
