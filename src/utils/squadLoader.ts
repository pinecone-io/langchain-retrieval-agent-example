import * as dfd from "danfojs-node";
import { dataFrameFromURL } from "./dataLoader.js";

const url =
  "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json";


function dropDuplicates(df: dfd.DataFrame, columnName: string): dfd.DataFrame {
  // Get the column as a series
  const series = df[columnName];

  // Drop duplicates from the series
  const cleanedSeries = series.dropDuplicates();

  // Create a new array filled with NaN for the length of the original DataFrame
  const filledValues = new Array(df.shape[0]).fill(NaN);

  // Replace the beginning of the filledValues array with the cleaned values
  // eslint-disable-next-line no-plusplus
  for (let i = 0; i < cleanedSeries.values.length; i++) {
    filledValues[i] = cleanedSeries.values[i];
  }

  // Create a new dataframe with the filled series
  const newDfData: Record<string, (string | number | boolean)[]> = {};
  for (const colName of df.columns) {
    if (colName === columnName) {
      newDfData[colName] = filledValues;
    } else {
      // For all the other columns, just copy the data over
      newDfData[colName] = df[colName].values;
    }
  }

  // Create a new DataFrame
  const newDf = new dfd.DataFrame(newDfData);

  // Drop rows containing NaN values
  newDf.dropNa({ axis: 1, inplace: true });

  return newDf;
}

const loadSquad = async (): Promise<dfd.DataFrame> => {
  const df: dfd.DataFrame = await dataFrameFromURL(
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

  df.rename({ "paragraphs.context": "context" }, { inplace: true });
  df.rename({ "paragraphs.qas.id": "id" }, { inplace: true });
  df.rename({ "paragraphs.qas.question": "question" }, { inplace: true });
  df.rename({ "paragraphs.qas.answers.text": "answer" }, { inplace: true });

  const cleanDf = dropDuplicates(df, "context");
  return cleanDf;
};

interface SquadRecord {
  context: string;
  id: string;
  question: string;
  answer: string;
}

export { loadSquad, SquadRecord };