class DtPeDataPreprocessCsvArgs:

    def __init__(self, input_csv, output_csv):
        self.input = input_csv
        self.output = output_csv
        self.encoding = 'utf-8'
        self.sep = ','
