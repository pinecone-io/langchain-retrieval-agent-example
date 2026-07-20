import { Parser, transforms } from "json2csv";
import fetch from "cross-fetch";
import { parseCSV, dropDuplicatesBy, type CSVRow } from "./csv.js";

const { unwind, flatten } = transforms;

const jsonToCSV = async (
  url: string,
  fields: string[],
  unwindFieldsPaths: string[]
): Promise<string> => {
  const response = await fetch(url);
  const { data } = await response.json();

  const csvTransforms = [
    unwind({ paths: [...unwindFieldsPaths] }),
    flatten({ objects: true, arrays: true }),
  ];

  const json2csvParser = new Parser({ fields, transforms: csvTransforms });
  return json2csvParser.parse(data);
};

// Fetches a nested JSON document, flattens it into CSV rows via json2csv's
// unwind/flatten transforms, and parses those rows straight into memory
// (previously this wrote the CSV to a temp file and read it back into a
// danfojs-node DataFrame; that dependency is gone, so is the temp file).
const rowsFromURL = async (
  url: string,
  fields: string[],
  unwindFieldsPaths: string[]
): Promise<CSVRow[]> => {
  const csv = await jsonToCSV(url, fields, unwindFieldsPaths);
  return parseCSV(csv);
};

export { rowsFromURL, dropDuplicatesBy };
