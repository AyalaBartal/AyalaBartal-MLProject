from src.specific.dt.trainer.pe_dt_train_report_args import DtPeTrainReportArgs
from src.specific.dt.trainer.args_validator import ArgsValidator
from src.specific.dt.trainer.pe_dt_train_output_mapper import DtPeTrainOutputMapper
from src.specific.dt.trainer.pe_dt_train_output_writer import DtPeTrainOutputWriter
from src.specific.dt.trainer.pe_dt_train_result import DtPeTrainResult


class DtPeTrainWriter:

    def __init__(self, mapper: DtPeTrainOutputMapper, writer: DtPeTrainOutputWriter):
        ArgsValidator.require_type_not_none(mapper, DtPeTrainOutputMapper, "mapper")
        ArgsValidator.require_type_not_none(writer, DtPeTrainOutputWriter, "writer")
        self.mapper = mapper
        self.writer = writer

    def write_output(self, args: DtPeTrainReportArgs, result: DtPeTrainResult):
        feature_names = self.mapper.get_feature_columns_list(result)
        dt_model = result.dt_model

        self.writer.write_text_to_text_file(args.out_report_md, self.mapper.get_end_message(result))
        self.writer.write_model_to_joblib_file(args.out_model_joblib, dt_model)
        self.writer.write_list_to_json_file(args.out_schema_json, feature_names)
        self.writer.write_object_to_json_file(args.out_report_json, self.mapper.get_report(args, result))
        self.writer.write_dt_model_to_graphviz_dot_file(args.model_output_dot, feature_names, dt_model)
        self.writer.write_feature_importance_to_csv(args.feature_importance_csv, feature_names, dt_model)
