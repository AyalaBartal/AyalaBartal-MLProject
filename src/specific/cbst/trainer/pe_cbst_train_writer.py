from src.common.validator import ArgsValidator
from src.specific.cbst.trainer.pe_cbst_train_report_args import CbstPeTrainReportArgs
from src.specific.cbst.trainer.pe_cbst_output_mapper import CbstPeOutputMapper
from src.specific.cbst.trainer.pe_cbst_output_writer import CbstPeOutputWriter
from src.specific.cbst.trainer.pe_cbst_train_result import CbstPeTrainResult


class CbstPeTrainWriter:
    """Write trained model and training reports."""

    def __init__(self, mapper: CbstPeOutputMapper, writer: CbstPeOutputWriter):
        ArgsValidator.require_type_not_none(mapper, CbstPeOutputMapper, "mapper")
        ArgsValidator.require_type_not_none(writer, CbstPeOutputWriter, "writer")
        self.mapper = mapper
        self.writer = writer

    def write_output(self, args: CbstPeTrainReportArgs, result: CbstPeTrainResult):
        """Write all training outputs: model, metrics, schema, and feature importance."""
        feature_names = self.mapper.get_feature_columns_list(result)
        cbst_model = result.cbst_model

        self.writer.write_text_to_text_file(args.out_report_md, self.mapper.get_end_message(result))
        self.writer.write_model_to_joblib_file(args.out_model_joblib, cbst_model)
        self.writer.write_list_to_json_file(args.out_schema_json, feature_names)
        self.writer.write_object_to_json_file(args.out_report_json, self.mapper.get_report(args, result))
        self.writer.write_feature_importance_to_csv(args.feature_importance_csv, feature_names, cbst_model)
