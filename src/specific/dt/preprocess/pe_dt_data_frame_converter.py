import numpy as np
import pandas as pd
from collections import Counter


class DtPeDataFrameConverter:

    # Convert an arrays of values to numbers and handle bad or missing data.
    @staticmethod
    def safe_num(value):
        # coerce: If conversion fails → return NaN instead of crashing.
        # fillna(0): replace NaN or missing data with 0.
        return pd.to_numeric(value, errors='coerce').fillna(0)

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
