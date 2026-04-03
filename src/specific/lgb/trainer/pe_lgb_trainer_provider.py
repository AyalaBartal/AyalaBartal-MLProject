from src.specific.lgb.trainer.pe_lgb_report_trainer import LgbPeReportTrainer
from src.specific.lgb.trainer.pe_lgb_data_trainer import LgbPeDataTrainer
from src.specific.lgb.trainer.pe_lgb_model_trainer import LgbPeModelTrainer
from src.specific.lgb.trainer.pe_lgb_logic_trainer import LgbPeLogicTrainer
from src.specific.lgb.trainer.pe_lgb_train_algo_args import LgbPeTrainAlgoArgs
from src.specific.lgb.trainer.pe_lgb_train_report_args import LgbPeTrainReportArgs
from src.specific.lgb.trainer.pe_lgb_io_trainer import LgbPeIoTrainer
from src.specific.lgb.trainer.pe_lgb_train_writer import LgbPeTrainWriter
from src.specific.lgb.trainer.pe_lgb_output_mapper import LgbPeOutputMapper
from src.specific.lgb.trainer.pe_lgb_output_writer import LgbPeOutputWriter


class LgbPeTrainerProvider:
    """Dependency injection factory for LightGBM trainer components."""

    @staticmethod
    def get_io_trainer():
        """Create fully configured I/O trainer with all dependencies."""
        model_trainer = LgbPeModelTrainer()
        data_trainer = LgbPeDataTrainer()
        report_trainer = LgbPeReportTrainer(data_trainer, model_trainer)
        logic_trainer = LgbPeLogicTrainer(data_trainer, model_trainer, report_trainer)
        lgb_output_mapper = LgbPeOutputMapper()
        output_writer = LgbPeOutputWriter()
        lgb_writer = LgbPeTrainWriter(lgb_output_mapper, output_writer)
        return LgbPeIoTrainer(data_trainer, logic_trainer, lgb_writer)
