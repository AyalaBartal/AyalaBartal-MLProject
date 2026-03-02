import os
import csv


class CsvIoValidate:
    """
    Validate that a file:
    - Exists
    - Is under max size
    - Is a text file (UTF-8)
    - Is a valid CSV
    - Contains only text values
    - Has exactly the expected ordered headers
    """

    @staticmethod
    def validate_csv_file(
            file_path: str,
            expected_headers: dict[int, str],
            max_size_bytes: int = 5 * 1024 * 1024  # default 50MB
    ) -> bool:

        # Check file exists
        if not os.path.isfile(file_path):
            raise ValueError("File does not exist")

        if not os.access(file_path, os.R_OK):
            raise ValueError("File exist but not readable")

        # Check file size
        size = os.path.getsize(file_path)
        if size > max_size_bytes:
            raise ValueError(f"File exceeds max size of {max_size_bytes} bytes")

        #  Try opening as UTF-8 text (reject binary)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                f.read(1024)  # Try reading sample
        except UnicodeDecodeError:
            raise ValueError("File is not a valid UTF-8 text file")

        # Validate CSV structure
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                headers = next(reader)
            except StopIteration:
                raise ValueError("CSV file is empty")

            #  Validate headers match EXACT order
            count_headers = len(headers)
            for index in expected_headers.keys():
                if index >= count_headers :
                    raise ValueError("Invalid header index: index {} vs count headers {}}".format(index, count_headers))
                header = headers[index]
                expected_header = expected_headers[index]
                if expected_header != header:
                    error = "Invalid header in index {} expected {} actual {}".format(index, expected_header, header)
                    raise ValueError(error)

            # Validate all rows contain only text
            for row_number, row in enumerate(reader, start=2):
                for col in row:
                    if not isinstance(col, str):
                        raise ValueError("Non-text value found at row {}".format(row_number))

        return True

