import json, math, re
import numpy as np
import pandas as pd

class DtPeDataConverter:

    @staticmethod
    def _parse_listish(s):
        if s is None or (isinstance(s, float) and math.isnan(s)): return []
        t = str(s).strip()
        try:
            if t.startswith('[') and t.endswith(']'): return [str(u) for u in json.loads(t) if u is not None]
        except Exception:
            pass
        return [w for w in re.split(r"[\s,;|]+", t) if w]

    @staticmethod
    def _clean_dll(ts):
        out = []
        for t in ts:
            t = str(t).lower().strip().strip('"\'');
            t = t.split('\\')[-1].split('/')[-1]
            if t.endswith('.dll'): t = t[:-4]
            if t: out.append(t)
        return out

    @staticmethod
    def _clean_api(ts):
        out = []
        for t in ts:
            t = str(t).lower().strip().strip('"\'');
            func = t.split('!', 1)[1] if '!' in t else t
            func = re.sub(r"[^a-z0-9_]+", " ", func);
            out.extend([w for w in func.split() if len(w) >= 2])
        return out

    @staticmethod
    def _clean_ident(v):
        if v is None or (isinstance(v, float) and math.isnan(v)): return []
        s = re.sub(r"[^a-z0-9_]+", " ", str(v).lower());
        return [w for w in s.split() if len(w) >= 2]

    @staticmethod
    def _safe_num(s):
        return pd.to_numeric(s, errors='coerce').fillna(0)

    @staticmethod
    def _expand_bits2(series, n_bits, prefix):
        x = pd.to_numeric(series, errors='coerce').fillna(0).astype(np.uint64)
        return pd.DataFrame({f"{prefix}_b{i}": ((x >> i) & 1).astype(np.int8) for i in range(n_bits)}, index=series.index)

    @staticmethod
    def _expand_bits(series, n_bits, prefix):
        x = (
            pd.to_numeric(series, errors='coerce')
            .fillna(0)
            .astype(np.uint64)
            .to_numpy()  # <-- convert to numpy array
        )

        return pd.DataFrame(
            {f"{prefix}_b{i}": ((x >> i) & 1).astype(np.int8) for i in range(n_bits)},
            index=series.index
        )

    @staticmethod
    def _to_dt(s):
        return pd.to_datetime(s, errors='coerce', utc=True)

    @staticmethod
    def _parse_tds(c):
        rn = pd.to_numeric(c, errors='coerce');
        an = ((rn == 0) | (rn == 0xFFFFFFFF)).fillna(False).astype(np.int8)
        dt = pd.to_datetime(rn, unit='s', errors='coerce', utc=True);
        m = dt.isna()
        if m.any(): dt = dt.where(~m, pd.to_datetime(c.astype(str), errors='coerce', utc=True))
        return dt, an

    @staticmethod
    def _dt_parts(dt, p):
        out = pd.DataFrame(index=dt.index);
        out[f"{p}_year"] = dt.dt.year.fillna(0).astype(int)
        out[f"{p}_month"] = dt.dt.month.fillna(0).astype(int)
        out[f"{p}_dow"] = dt.dt.dayofweek.fillna(0).astype(int)
        return out

    @staticmethod
    def _ratio(df, a, b):
        A = pd.to_numeric(df.get(a), errors='coerce')
        B = pd.to_numeric(df.get(b), errors='coerce')
        return (A / (B.replace(0, np.nan))).fillna(0)

    @staticmethod
    def _topk(series, tokenizer, k, prefix):
        from collections import Counter
        toks = []
        ctr = Counter()
        for v in series.tolist():
            t = tokenizer(v)
            toks.append(t)
            ctr.update(t)
        vocab = [w for w, _ in ctr.most_common(max(0, k))] if k > 0 else []
        idx = {w: i for i, w in enumerate(vocab)};
        M = np.zeros((len(series), len(vocab)), dtype=np.int32)
        for i, ts in enumerate(toks):
            for t in ts:
                j = idx.get(t)
                if j is not None: M[i, j] += 1
        return pd.DataFrame(M, columns=[f"{prefix}_{w}" for w in vocab], index=series.index)
