// Minimal CSV reader. This project only ever used `danfojs-node` to hold the
// flattened SQuAD rows and drop duplicate contexts, so we inline that tiny
// piece here and drop the dependency (which pulled in TensorFlow.js Node
// bindings and a large, vulnerable transitive tree: tar, xlsx, qs,
// tough-cookie, uuid).
//
// The parser follows RFC 4180: fields may be wrapped in double quotes, in
// which case they can contain commas, CR/LF and escaped quotes (`""`). This
// matters because SQuAD's context passages contain all three.

export type CSVRow = Record<string, string>;

// Splits raw CSV text into rows of string cells, honouring quoted fields.
function parseRows(text: string): string[][] {
  const rows: string[][] = [];
  let field = "";
  let row: string[] = [];
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];

    if (inQuotes) {
      if (char === '"') {
        if (text[i + 1] === '"') {
          // Escaped quote: consume both characters, keep one.
          field += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        field += char;
      }
    } else if (char === '"') {
      inQuotes = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      // Ignore blank lines rather than emitting an all-empty row.
      if (field !== "" || row.length > 0) {
        row.push(field);
        rows.push(row);
      }
      field = "";
      row = [];
    } else if (char !== "\r") {
      // A bare CR (or the CR of a CRLF) outside quotes is a line-ending artefact.
      field += char;
    }
  }

  // Flush a trailing row that was not terminated by a newline.
  if (field !== "" || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  return rows;
}

// Parses CSV text into row objects keyed by the header row.
export function parseCSV(text: string): CSVRow[] {
  const rows = parseRows(text);
  if (rows.length === 0) {
    return [];
  }

  const [header, ...body] = rows;
  return body.map((cells) => {
    const record: CSVRow = {};
    header.forEach((key, i) => {
      record[key] = cells[i] ?? "";
    });
    return record;
  });
}

// Keeps only the first row seen for each distinct value of `columnName`,
// mirroring the behaviour of the danfojs `dropDuplicates` helper this
// pipeline previously relied on.
export function dropDuplicatesBy<T>(rows: T[], columnName: keyof T): T[] {
  const seen = new Set<unknown>();
  return rows.filter((row) => {
    const key = row[columnName];
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}
