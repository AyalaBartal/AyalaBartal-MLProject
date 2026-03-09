from src.specific.dt.preprocess import DtPeDataPreprocessArgs
from src.specific.dt.preprocess.pe_dt_data_converter import DtPeDataConverter
from src.specific.dt.preprocess.pe_dt_data_transformer import DtPeDataTransformer
from src.specific.dt.preprocess.pe_dt_preprocess_mapper import DtPePreprocessMapper


class DtPePreprocessorProvider:

    @staticmethod
    def get_mapper():
        converter = DtPeDataConverter()
        dt_pe_data_transformer = DtPeDataTransformer(converter)
        dt_pe_data_preprocess_args = DtPeDataPreprocessArgs("input.csv", "output.csv")
        dt_pe_data_preprocessor = DtPePreprocessMapper(dt_pe_data_preprocess_args, dt_pe_data_transformer)
        return dt_pe_data_preprocessor
