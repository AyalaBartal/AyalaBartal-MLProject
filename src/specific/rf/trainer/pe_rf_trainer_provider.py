from src.specific.rf.trainer.pe_rf_report_trainer import RfPeReportTrainer
from src.specific.rf.trainer.pe_rf_data_trainer import RfPeDataTrainer
from src.specific.rf.trainer.pe_rf_model_trainer import RfPeModelTrainer
from src.specific.rf.trainer.pe_rf_logic_trainer import RfPeLogicTrainer
from src.specific.rf.trainer.pe_rf_train_algo_args import RfPeTrainAlgoArgs
from src.specific.rf.trainer.pe_rf_train_report_args import RfPeTrainReportArgs
from src.specific.rf.trainer.pe_rf_io_trainer import RfPeIoTrainer
from src.specific.rf.trainer.pe_rf_train_writer import RfPeTrainWriter
from src.specific.rf.trainer.pe_rf_train_output_mapper import RfPeTrainOutputMapper
from src.specific.rf.trainer.pe_rf_train_output_writer import RfPeTrainOutputWriter


class RfPeTrainerProvider:

    @staticmethod
    def get_io_trainer():
        model_trainer = RfPeModelTrainer()
        data_trainer = RfPeDataTrainer()
        report_trainer = RfPeReportTrainer(data_trainer, model_trainer)
        logic_trainer = RfPeLogicTrainer(data_trainer, model_trainer, report_trainer)
        rf_output_mapper = RfPeTrainOutputMapper()
        output_writer = RfPeTrainOutputWriter()
        rf_writer = RfPeTrainWriter(rf_output_mapper, output_writer)
        return RfPeIoTrainer(data_trainer, logic_trainer, rf_writer)
