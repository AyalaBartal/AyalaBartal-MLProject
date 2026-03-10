from src.specific.dt.preprocess.pe_dt_string_converter import DtPeStringConverter
from src.specific.dt.preprocess.pe_dt_list_converter import DtPeListConverter
from src.specific.dt.preprocess.pe_dt_data_frame_converter import DtPeDataFrameConverter

from src.specific.dt.preprocess.pe_dt_preprocess_map_args import DtPeDataPreprocessMapArgs
from src.specific.dt.preprocess.column_transformer_registry import ColumnTransformerRegistry
from src.specific.dt.preprocess.pe_dt_data_transformer import DtPeDataTransformer
from src.specific.dt.preprocess.pe_dt_preprocess_mapper import DtPePreprocessMapper


class DtPePreprocessorProvider:

    @staticmethod
    def get_mapper():
        args = DtPeDataPreprocessMapArgs()
        c_str = DtPeStringConverter()
        c_list = DtPeListConverter()
        c_df = DtPeDataFrameConverter()
        registry = ColumnTransformerRegistry(c_str, c_list, c_df, args)
        dt_pe_data_transformer = DtPeDataTransformer(registry)
        dt_pe_data_preprocessor = DtPePreprocessMapper(args, dt_pe_data_transformer)
        return dt_pe_data_preprocessor

    @staticmethod
    def get_transformer():
        args = DtPeDataPreprocessMapArgs()
        c_str = DtPeStringConverter()
        c_list = DtPeListConverter()
        c_df = DtPeDataFrameConverter()
        registry = ColumnTransformerRegistry(c_str, c_list, c_df, args)
        return DtPeDataTransformer(registry)
