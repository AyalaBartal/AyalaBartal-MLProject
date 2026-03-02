import csv
import io
from typing import List, Dict, Any


class CsvUtil:

    @staticmethod
    # Convert csv content, list of dictionaries, into CSV-formatted bytes.
    def csv_data_to_bytes(input_data: List[Dict]) -> bytes:
        if not input_data:
            return b""

        headers = input_data[0].keys()

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        writer.writerows(input_data)

        return output.getvalue().encode("utf-8")

    @staticmethod
    # Return a map of column index -> header name.
    def map_index_to_headers(rows: List[Dict[str, Any]]) -> Dict[int, str]:
        if not rows:
            return {}
        headers = list(rows[0].keys())
        return {index: header for index, header in enumerate(headers)}

    @staticmethod
    # Extract a single column by index from list[dict] CSV data. Returns list[dict] containing only that column.
    def extract_column_by_index(
            rows: List[Dict],
            column_index: int
    ) -> List[Dict]:

        if not rows:
            return []

        if column_index < 0 or column_index >= len(rows[0].keys()):
            raise IndexError("Column index out of range {}".format(column_index))

        headers = list(rows[0].keys())
        header = headers[column_index]
        return [{header: row[header]} for row in rows]

    @staticmethod
    # Rename a column header (by index) across all rows, preserving column order.
    # Only modifies the header (dict key), not the values.
    # Modifies the input in-place and return None.

    def modify_column_by_index(
            rows: List[Dict[str, Any]],
            column_index: int,
            new_header: str
    ) -> None:

        if not rows:
            return

        if column_index < 0 or column_index >= len(rows[0].keys()):
            raise IndexError("Column index out of range {}".format(column_index))

        headers = list(rows[0].keys())

        if new_header == headers[column_index]:
            return

        if new_header in headers:
            raise ValueError(f"New header '{new_header}' already exists")

        # New header list with same order, just renamed at the index
        new_headers = headers.copy()
        new_headers[column_index] = new_header

        # Rebuild every row using the original order of values
        for i in range(len(rows)):
            row = rows[i]

            # Use original headers order to fetch values deterministically
            values = [row.get(h) for h in headers]

            # Build new row with preserved order and renamed header
            rows[i] = dict(zip(new_headers, values))