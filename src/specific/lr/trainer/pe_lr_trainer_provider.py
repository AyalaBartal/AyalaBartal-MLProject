from src.specific.lr.trainer.pe_lr_data_trainer import LrPeDataTrainer
from src.specific.lr.trainer.pe_lr_model_trainer import LrPeModelTrainer
from src.specific.lr.trainer.pe_lr_report_trainer import LrPeReportTrainer
from src.specific.lr.trainer.pe_lr_logic_trainer import LrPeLogicTrainer
from src.specific.lr.trainer.pe_lr_train_output_mapper import LrPeTrainOutputMapper
from src.specific.lr.trainer.pe_lr_train_output_writer import LrPeTrainOutputWriter
from src.specific.lr.trainer.pe_lr_train_writer import LrPeTrainWriter


class LrPeTrainerProvider:

    @staticmethod
    def get_data_trainer():
        return LrPeDataTrainer()

    @staticmethod
    def get_model_trainer():
        return LrPeModelTrainer()

    @staticmethod
    def get_report_trainer():
        return LrPeReportTrainer()

    @staticmethod
    def get_logic_trainer():
        data_trainer = LrPeTrainerProvider.get_data_trainer()
        model_trainer = LrPeTrainerProvider.get_model_trainer()
        report_trainer = LrPeTrainerProvider.get_report_trainer()
        return LrPeLogicTrainer(data_trainer, model_trainer, report_trainer)

    @staticmethod
    def get_output_mapper():
        return LrPeTrainOutputMapper()

    @staticmethod
    def get_output_writer():
        return LrPeTrainOutputWriter()

    @staticmethod
    def get_train_writer():
        return LrPeTrainWriter(LrPeTrainerProvider.get_output_mapper(), LrPeTrainerProvider.get_output_writer())
