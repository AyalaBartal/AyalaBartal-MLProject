from src.specific.ml.trainer.pe_ml_data_trainer import MlPeDataTrainer
from src.specific.ml.trainer.pe_ml_model_trainer import MlPeModelTrainer
from src.specific.ml.trainer.pe_ml_report_trainer import MlPeReportTrainer
from src.specific.ml.trainer.pe_ml_logic_trainer import MlPeLogicTrainer
from src.specific.ml.trainer.pe_ml_train_output_mapper import MlPeTrainOutputMapper
from src.specific.ml.trainer.pe_ml_train_output_writer import MlPeTrainOutputWriter
from src.specific.ml.trainer.pe_ml_train_writer import MlPeTrainWriter


class MlPeTrainerProvider:

    @staticmethod
    def get_data_trainer():
        return MlPeDataTrainer()

    @staticmethod
    def get_model_trainer():
        return MlPeModelTrainer()

    @staticmethod
    def get_report_trainer():
        return MlPeReportTrainer()

    @staticmethod
    def get_logic_trainer():
        data_trainer = MlPeTrainerProvider.get_data_trainer()
        model_trainer = MlPeTrainerProvider.get_model_trainer()
        report_trainer = MlPeTrainerProvider.get_report_trainer()
        return MlPeLogicTrainer(data_trainer, model_trainer, report_trainer)

    @staticmethod
    def get_output_mapper():
        return MlPeTrainOutputMapper()

    @staticmethod
    def get_output_writer():
        return MlPeTrainOutputWriter()

    @staticmethod
    def get_train_writer():
        return MlPeTrainWriter(MlPeTrainerProvider.get_output_mapper(), MlPeTrainerProvider.get_output_writer())
