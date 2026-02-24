import pandas as pd

from src.specific.dt.preprocess.pe_dt_preprocess_mapper import DtPePreprocessMapper


class DtPeCsvPreprocessMapper:


    def __init__(self, transformer):
        self.dt_pe_data_transformer = transformer

    def map(self, args):
        mapper = DtPePreprocessMapper(args, self.dt_pe_data_transformer)

        # 1 Read entire csv into data1
        data1 = pd.read_csv(args.input, sep=args.sep, encoding=args.encoding)

        # 2 pre_process -> transform data1 to data2
        data2 = mapper.map(data1)

        # 3 write data2 into output file
        data2.to_csv(args.output, index=False)
        output_rows_count = len(data2)
        output_shape = data2.shape[1]
        print(f"Saved {output_rows_count} rows × {output_shape} cols to {args.output}")