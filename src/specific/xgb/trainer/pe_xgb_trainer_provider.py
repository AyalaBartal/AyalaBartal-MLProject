from src.specific.xgb.trainer.pe_xgb_report_trainer import XgbPeReportTrainer
from src.specific.xgb.trainer.pe_xgb_data_trainer import XgbPeDataTrainer
from src.specific.xgb.trainer.pe_xgb_model_trainer import XgbPeModelTrainer
from src.specific.xgb.trainer.pe_xgb_logic_trainer import XgbPeLogicTrainer
from src.specific.xgb.trainer.pe_xgb_train_algo_args import XgbPeTrainAlgoArgs
from src.specific.xgb.trainer.pe_xgb_train_report_args import XgbPeTrainReportArgs
from src.specific.xgb.trainer.pe_xgb_io_trainer import XgbPeIoTrainer
from src.specific.xgb.trainer.pe_xgb_train_writer import XgbPeTrainWriter
from src.specific.xgb.trainer.pe_xgb_output_mapper import XgbPeOutputMapper
from src.specific.xgb.trainer.pe_xgb_output_writer import XgbPeOutputWriter


class XgbPeTrainerProvider:
    """Dependency injection factory for XGBoost trainer components."""

    @staticmethod
    def get_io_trainer():
        """Create fully configured I/O trainer with all dependencies."""
        model_trainer = XgbPeModelTrainer()
        data_trainer = XgbPeDataTrainer()
        report_trainer = XgbPeReportTrainer(data_trainer, model_trainer)
        logic_trainer = XgbPeLogicTrainer(data_trainer, model_trainer, report_trainer)
        xgb_output_mapper = XgbPeOutputMapper()
        output_writer = XgbPeOutputWriter()
        xgb_writer = XgbPeTrainWriter(xgb_output_mapper, output_writer)
        return XgbPeIoTrainer(data_trainer, logic_trainer, xgb_writer)
