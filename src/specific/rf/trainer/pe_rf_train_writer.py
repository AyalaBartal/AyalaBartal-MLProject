from src.common.validator import ArgsValidator
from src.specific.rf.trainer.pe_rf_train_report_args import RfPeTrainReportArgs
from src.specific.rf.trainer.pe_rf_train_output_mapper import RfPeTrainOutputMapper
from src.specific.rf.trainer.pe_rf_train_output_writer import RfPeTrainOutputWriter
from src.specific.rf.trainer.pe_rf_train_result import RfPeTrainResult


class RfPeTrainWriter:

    def __init__(self, mapper: RfPeTrainOutputMapper, writer: RfPeTrainOutputWriter):
        ArgsValidator.require_type_not_none(mapper, RfPeTrainOutputMapper, "mapper")
        ArgsValidator.require_type_not_none(writer, RfPeTrainOutputWriter, "writer")
        self.mapper = mapper
        self.writer = writer

    def write_output(self, args: RfPeTrainReportArgs, result: RfPeTrainResult):
        feature_names = self.mapper.get_feature_columns_list(result)
        rf_model = result.rf_model

        self.writer.write_text_to_text_file(args.out_report_md, self.mapper.get_end_message(result))
        self.writer.write_model_to_joblib_file(args.out_model_joblib, rf_model)
        self.writer.write_list_to_json_file(args.out_schema_json, feature_names)
        self.writer.write_object_to_json_file(args.out_report_json, self.mapper.get_report(args, result))
        self.writer.write_feature_importance_to_csv(args.feature_importance_csv, feature_names, rf_model)
