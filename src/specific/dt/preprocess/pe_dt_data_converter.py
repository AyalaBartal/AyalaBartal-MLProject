import json
import math
import re
import numpy as np
import pandas as pd
from collections import Counter


class DtPeDataConverter:

    # Convert a single  value (number, json, sentence) into a list of simple words.
    @staticmethod
    def parse_listish(value):
        text = str(value).strip()

        # Case 1: value is empty: None and NaN. NaN is float.
        if value is None or text.lower() in {"nan", "none", ""}:
            return []

        # Case 2: value is an empty json.
        if text == "" or text == "[]" or text == "{}":
            return []

        # Case 3: value is a non-empty json: ["a","b"] or [1,2,3], [{"a":1}, {"b":2}]
        if text.startswith("[") and text.endswith("]"):
            json_data = json.loads(text)
            return [str(item) for item in json_data if item is not None]

        # Case 4: default fallback, split by common separators and filter out empty words
        words = re.split(r"[\s,;|]+", text)
        return [w for w in words if w]

    # Normalizes DLL names from a sequence (ts) to a list of DLL names without paths and without the .dll extension.
    @staticmethod
    def clean_dll(input_dlls):
        output_dlls = []
        for value in input_dlls:
            # Convert into String, lower case,remove surrounding whitespace and quotes.
            value = str(value).lower().strip().strip('"\'')
            # Remove file path (Windows and Linux)
            value = value.split('\\')[-1].split('/')[-1]
            # Remove suffix .dll, if exists, by removing the last 4 chars.
            if value.endswith('.dll'):
                value = value[:-4]
            # Skip empty values and add to output all non-empty values.
            if value:
                output_dlls.append(value)
        return output_dlls

    # Normalizes API/methods paths. Example: [ "kernel32!CreateFileW" ] --> [createfilew]
    @staticmethod
    def clean_api(input_methods):
        output_methods = []
        for value in input_methods:
            # Convert into String, lower case,remove surrounding whitespace and quotes.
            value = str(value).lower().strip().strip('"\'')
            # Extracts from value only the function name: kernel32!createfilew --> createfilew
            func = value.split('!', 1)[1] if '!' in value else value
            # Replace special characters with space: create-file@1 --> create file
            func = re.sub(r"[^a-z0-9_]+", " ", func)
            # Split sentence by space into words: "create file w" -> ["create", "file", "w"]
            # Filter out short words with less than three chars. Add only and all long words to output.
            output_methods.extend([w for w in func.split() if len(w) >= 2])
        return output_methods

    # Normalizes an identifier string and converts it into words
    @staticmethod
    def clean_ident(value):
        # If value is empty: None and NaN. NaN is float. Then return empty array.
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return []
        # Convert into String, lower case, Replace non-identifier characters with space.
        s = re.sub(r"[^a-z0-9_]+", " ", str(value).lower())
        # Split sentence by space into words: "create file w" -> ["create", "file", "w"]
        # Filter out short words with less than three chars. Add only and all long words to output. 1
        return [w for w in s.split() if len(w) >= 2]

    # Convert an arrays of values to numbers and handle bad or missing data.
    @staticmethod
    def safe_num(value):
        # coerce: If conversion fails → return NaN instead of crashing.
        # fillna(0): replace NaN or missing data with 0.
        return pd.to_numeric(value, errors='coerce').fillna(0)

    # Converts an integer column into multiple binary feature columns, one column per bits position.
    @staticmethod
    def expand_bits2(series, n_bits, prefix):
        # coerce: If conversion fails → return NaN instead of crashing.
        # fillna(0): replace NaN or missing data with 0.
        # astype convert int64 that supports up to 64 bits.
        x = pd.to_numeric(series, errors='coerce').fillna(0).astype(np.uint64)
        # Convert number to bits.
        # Shift 1 bit
        # Convert from int64 to int8 This saves memory (1 byte not 8 bytes).
        bits = {f"{prefix}_b{i}": ((x >> i) & 1).astype(np.int8) for i in range(n_bits)}
        return pd.DataFrame(bits, index=series.index)

    # Expands a numeric bitmask column into multiple binary feature columns.
    # Instead of one number, the method creates separate columns for each bit.
    @staticmethod
    def expand_bits(series, n_bits, prefix):
        x = (
            # Convert into number. Invalid data become zero.
            pd.to_numeric(series, errors='coerce')
            # Replace missing values with zero
            .fillna(0)
            # Convert to int64
            .astype(np.uint64)
            # Convert to NumPy array
            .to_numpy()  # <-- convert to numpy array
        )

        return pd.DataFrame(
            # Build DataFrame columns for each bit in number.
            # Shift 1 bit right. Extract the last bit. Convert int64 to int8.
            {f"{prefix}_b{i}": ((x >> i) & 1).astype(np.int8) for i in range(n_bits)},
            index=series.index
        )

    # Converts a value (or column) into a datetime using pandas.
    @staticmethod
    def to_dt(value):
        # If a value cannot be converted to a date → return NaT instead of throwing an error.
        # Use UTC timezone aware timestamps.
        return pd.to_datetime(value, errors='coerce', utc=True)

    # Convert timestamp into columns (year, month, dow)
    # Input that may contain either Unix timestamps or special values.
    # Returns two outputs: dt → parsed datetime and an → anomaly flag
    @staticmethod
    def parse_tds(c):
        # Convert input to numeric.  If a value cannot be converted → return NaT instead of throwing an error.
        rn = pd.to_numeric(c, errors='coerce')
        # Detect anomaly values. This creates an anomaly flag column.
        # fillna(False) ensures NaN values result in False and not NaN booleans
        # astype(np.int8) -> convert to int8
        an = ((rn == 0) | (rn == 0xFFFFFFFF)).fillna(False).astype(np.int8)
        # Convert numeric timestamp, in seconds, to datetime. Use UTC timezone aware timestamps.
        #  If a value cannot be converted → return NaT instead of throwing an error.
        dt = pd.to_datetime(rn, unit='s', errors='coerce', utc=True)
        # m = True where parsing failed
        m = dt.isna()
        # Replace failed rows. Try string date parsing on rows that failed Unix time parsing.
        # If a value cannot be converted → return NaT instead of throwing an error. Use UTC timezone aware timestamps.
        if m.any():
            dt = dt.where(~m, pd.to_datetime(c.astype(str), errors='coerce', utc=True))
        # two Series: dt → parsed datetime and an → anomaly flag
        return dt, an

    # Convert timestamp into columns (year, month, dow)
    @staticmethod
    def dt_parts(dt, p):
        # Create output DataFrame. It keeps the same index as the input series.
        out = pd.DataFrame(index=dt.index)
        # Extracts the year/month/dow as int from datetime. If datetime is invalid make it Nan.
        out[f"{p}_year"] = dt.dt.year.fillna(0).astype(int)
        out[f"{p}_month"] = dt.dt.month.fillna(0).astype(int)
        # dow -> day-of-week: Monday 0, Tuesday 1, Wednesday 2, Thursday 3, Friday 4, Saturday	5, Sunday
        out[f"{p}_dow"] = dt.dt.dayofweek.fillna(0).astype(int)
        return out

    # Computes a ratio between two DataFrame, df, columns: a and b.
    # Return Nan if a or b are invalid or if b is NaN.
    @staticmethod
    def ratio(df, a, b):
        a_value = pd.to_numeric(df.get(a), errors='coerce')
        b_value = pd.to_numeric(df.get(b), errors='coerce')
        return (a_value / (b_value.replace(0, np.nan))).fillna(0)

    # This method converts a text column into numeric ML features using the top K most common words.
    @staticmethod
    def topk(series, tokenizer, k, prefix):
        # 1) Calculate words array and word_counter map from word to its counter (times in text).
        #  counts how many times each word/token appear in text.
        #  Stores global word/token frequencies across the dataset.
        words = []
        word_counter = Counter()
        for v in series.tolist():
            token = tokenizer(v)
            words.append(token)
            word_counter.update(token)

        # 2) Select top-K vocabulary (assume k is not negative)
        k_positive = max(0, k)
        k_top = word_counter.most_common(k_positive)
        vocab = [w for w, _ in k_top] if k > 0 else []

        # 3) Calculate value for each row and new token binary column
        # Maps tokens to column positions.
        idx = {w: i for i, w in enumerate(vocab)}
        # Create feature matrix
        matrix = np.zeros((len(series), len(vocab)), dtype=np.int32)
        # Fill token counts
        for i, ts in enumerate(words):
            for token in ts:
                j = idx.get(token)
                if j is not None:
                    matrix[i, j] += 1

        # 4) Convert to DataFrame. Columns are renamed using an input prefix.
        return pd.DataFrame(matrix, columns=[f"{prefix}_{w}" for w in vocab], index=series.index)
