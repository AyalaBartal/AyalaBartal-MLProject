import pandas as pd

from src.specific.dt.preprocess import DtPeDataPreprocessCsvArgs


class DtPeCsvPreprocessMapper:

    def __init__(self, mapper):
        self.mapper = mapper

    def map(self, args: DtPeDataPreprocessCsvArgs):
        print("Start preprocess with input={} and output={}".format(args.input, args.output))
        # 1 Read entire csv into data1
        data1 = pd.read_csv(args.input, sep=args.sep, encoding=args.encoding)

        # 2 pre_process -> transform data1 to data2
        data2 = self.mapper.map(data1)

        # 3 write data2 into output file
        data2.to_csv(args.output, index=False)
        print("End preprocess with input={}, output={} and size={}".format(args.input, args.output, data2.shape))

