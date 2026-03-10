from src.specific.dt.preprocess.pe_dt_preprocess_map_args import DtPeDataPreprocessMapArgs
from src.specific.dt.preprocess.column_transformer_registry import ColumnTransformerRegistry
from src.specific.dt.preprocess.pe_dt_data_converter import DtPeDataConverter
from src.specific.dt.preprocess.pe_dt_data_transformer import DtPeDataTransformer
from src.specific.dt.preprocess.pe_dt_preprocess_mapper import DtPePreprocessMapper


class DtPePreprocessorProvider:

    @staticmethod
    def get_mapper():
        args = DtPeDataPreprocessMapArgs()
        converter = DtPeDataConverter()
        registry = ColumnTransformerRegistry(converter, args.k_dlls, args.k_apis, args.k_ident, args.bit_count)
        dt_pe_data_transformer = DtPeDataTransformer(registry)
        dt_pe_data_preprocessor = DtPePreprocessMapper(args, dt_pe_data_transformer)
        return dt_pe_data_preprocessor
