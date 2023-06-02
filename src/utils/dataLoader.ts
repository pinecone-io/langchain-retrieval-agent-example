/* eslint-disable import/no-extraneous-dependencies */
import * as dfd from "danfojs-node";
import { Parser, transforms } from "json2csv";
import fs from "fs";
import fetch from "cross-fetch";

const { unwind, flatten } = transforms;

const jsonToCSV = async (
  url: string,
  fields: string[],
  unwindFieldsPaths: string[]
): Promise<string> => {
  const response = await fetch(url);
  const { data } = await response.json();

  const topLevelData = data;
  const transforms = [unwind({ paths: [...unwindFieldsPaths] }), flatten({ objects: true, arrays: true })];

  const json2csvParser = new Parser({ fields, transforms });
  const csv = json2csvParser.parse(topLevelData);

  return csv;
};

const dataFrameFromURL = async (
  url: string,
  fields: string[],
  unwindFieldsPaths: string[]
): Promise<dfd.DataFrame> => {
  const csv = await jsonToCSV(url, fields, unwindFieldsPaths);
  // generate random file name
  const name = Math.random().toString(36).substring(7);

  const filePath = `./${name}.csv`;
  try {
    fs.writeFile(filePath, csv, (err) => {
      if (err) throw err;
    });
  } catch (err) {
    console.log(err);
  }

  const df: dfd.DataFrame = (await dfd.readCSV(filePath)) as dfd.DataFrame;

  // delete the file in try catch, asynchronously
  try {
    fs.unlinkSync(filePath);
  } catch (err) {
    console.error(err);
  }
  return df;
};

const dropDuplicates = (df: dfd.DataFrame, columnName: string): dfd.DataFrame => {
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
};

export { dataFrameFromURL, dropDuplicates };
