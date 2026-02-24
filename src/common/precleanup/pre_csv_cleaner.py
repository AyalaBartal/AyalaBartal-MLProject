import pandas as pd


class PeCsvPreprocessCleaner:


    def __init__(self, input_file, output_file, cleaner):
        self.input_file = input_file
        self.output_file = output_file
        self.cleaner = cleaner

    def pre_clean_csv(self):
        input_data = pd.read_csv(self.input_file)
        output_data = self.cleaner.clean(input_data)
        output_data.to_csv(self.output_file)
