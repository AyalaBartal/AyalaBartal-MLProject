from src.common.preprocessor.column_transformer_map_provider import ColumnTransformerMapProvider
from src.common.preprocessor.column_transformer_one_provider import ColumnTransformerOneProvider
from src.common.preprocessor.pe_dt_string_converter import DtPeStringConverter
from src.common.preprocessor.pe_dt_list_converter import DtPeListConverter
from src.common.preprocessor.pe_dt_data_frame_converter import DtPeDataFrameConverter

from src.common.preprocessor.pe_dt_preprocess_map_args import DtPeDataPreprocessMapArgs
from src.common.preprocessor.column_transformer_registry import ColumnTransformerRegistry
from src.common.preprocessor.pe_dt_data_transformer import DtPeDataTransformer
from src.common.preprocessor.pe_dt_preprocess_mapper import DtPePreprocessMapper


class DtPePreprocessorProvider:

    @staticmethod
    def get_mapper():
        args = DtPeDataPreprocessMapArgs()
        c_str = DtPeStringConverter()
        c_list = DtPeListConverter()
        c_df = DtPeDataFrameConverter()
        c_t_one_provider = ColumnTransformerOneProvider(c_str, c_list, c_df)
        c_t_map_provider = ColumnTransformerMapProvider(c_t_one_provider)
        registry = ColumnTransformerRegistry(c_t_map_provider, args)
        dt_pe_data_transformer = DtPeDataTransformer(registry)
        dt_pe_data_preprocessor = DtPePreprocessMapper(args, dt_pe_data_transformer)
        return dt_pe_data_preprocessor

    @staticmethod
    def get_transformer():
        args = DtPeDataPreprocessMapArgs()
        c_str = DtPeStringConverter()
        c_list = DtPeListConverter()
        c_df = DtPeDataFrameConverter()
        c_t_one_provider = ColumnTransformerOneProvider(c_str, c_list, c_df)
        c_t_map_provider = ColumnTransformerMapProvider(c_t_one_provider)
        registry = ColumnTransformerRegistry(c_t_map_provider, args)
        return DtPeDataTransformer(registry)
