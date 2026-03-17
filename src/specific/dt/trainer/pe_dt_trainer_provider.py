from src.specific.dt.trainer.pe_dt_report_trainer import DtPeReportTrainer
from src.specific.dt.trainer.pe_dt_data_trainer import DtPeDataTrainer
from src.specific.dt.trainer.pe_dt_model_trainer import DtPeModelTrainer
from src.specific.dt.trainer.pe_dt_logic_trainer import DtPeLogicTrainer
from src.specific.dt.trainer.pe_dt_train_algo_args import DtPeTrainAlgoArgs
from src.specific.dt.trainer.pe_dt_train_report_args import DtPeTrainReportArgs
from src.specific.dt.trainer.pe_dt_io_trainer import DtPeIoTrainer
from src.specific.dt.trainer.pe_dt_train_writer import DtPeTrainWriter
from src.specific.dt.trainer.pe_dt_train_output_mapper import DtPeTrainOutputMapper
from src.specific.dt.trainer.pe_dt_train_output_writer import DtPeTrainOutputWriter


class DtPeTrainerProvider:

    @staticmethod
    def get_io_trainer():
        model_trainer = DtPeModelTrainer()
        data_trainer = DtPeDataTrainer()
        report_trainer = DtPeReportTrainer(data_trainer, model_trainer)
        logic_trainer = DtPeLogicTrainer(data_trainer, model_trainer, report_trainer)
        dt_output_mapper = DtPeTrainOutputMapper()
        output_writer = DtPeTrainOutputWriter()
        dt_writer = DtPeTrainWriter(dt_output_mapper, output_writer)
        return DtPeIoTrainer(data_trainer, logic_trainer, dt_writer)
