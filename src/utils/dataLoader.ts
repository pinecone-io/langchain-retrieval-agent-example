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

  const filePath = `./data/${name}.csv`;
  try {
    fs.writeFile(filePath, csv, (err) => {
      if (err) throw err;
      // console.log("The file has been saved!");
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

export { dataFrameFromURL };
