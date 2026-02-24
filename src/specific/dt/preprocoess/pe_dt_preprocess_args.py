class DtPeDataPreprocessArgs:

    def __init__(self, input_csv, output_csv):
        self.input = input_csv
        self.output = output_csv
        self.label_col = 'label'
        self.k_ident = 100
        self.k_dlls = 100
        self.k_apis = 200
        self.bit_count = 16
        self.encoding = 'utf-8'
        self.label_col = 'Label'
        self.sep = ','
