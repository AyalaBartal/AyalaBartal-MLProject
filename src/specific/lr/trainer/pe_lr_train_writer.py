from src.specific.lr.trainer.pe_lr_train_output_mapper import LrPeTrainOutputMapper
from src.specific.lr.trainer.pe_lr_train_output_writer import LrPeTrainOutputWriter


class LrPeTrainWriter:

    def __init__(self, output_mapper: LrPeTrainOutputMapper, output_writer: LrPeTrainOutputWriter):
        self.output_mapper = output_mapper
        self.output_writer = output_writer

    def write(self, report, output_dir, train_result):
        mapped_output = self.output_mapper.map_report_to_output(report, train_result)
        model_path = self.output_writer.write_model(output_dir, mapped_output['model'])
        metrics_path = self.output_writer.write_metrics(output_dir, mapped_output['metrics'])
        schema_path = self.output_writer.write_feature_schema(output_dir, mapped_output['feature_schema'])
        return {
            'model_path': model_path,
            'metrics_path': metrics_path,
            'schema_path': schema_path
        }
