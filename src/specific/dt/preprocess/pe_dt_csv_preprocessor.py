import pandas as pd


class DtPeCsvPreprocessMapper:

    def __init__(self, mapper):
        self.mapper = mapper

    def map(self, input_file, output_file):
        print("Start preprocess with input={} and output={}".format(input_file, output_file))
        # 1 Read entire csv into data1
        data1 = pd.read_csv(input_file, sep=',', encoding='utf-8')

        # 2 pre_process -> transform data1 to data2
        data2 = self.mapper.map(data1)

        # 3 write data2 into output file
        data2.to_csv(output_file, index=False)
        print("End preprocess with input={} and output={} and size={}".format(input_file, output_file, data2.shape))
