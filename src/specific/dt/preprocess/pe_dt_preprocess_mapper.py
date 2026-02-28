import pandas as pd

class DtPePreprocessMapper:


    def __init__(self, args, transformer):
        self.args = args
        self.dt_pe_data_transformer = transformer

    def map(self, data1):
        args = self.args
        data2 = self.dt_pe_data_transformer.transform(data1, args.k_ident, args.k_dlls, args.k_apis, args.bit_count)
        if args.label_col and args.label_col in data1.columns:
            data2[args.label_col] = data1[args.label_col].values
        return data2
