from src.specific.cbst.trainer.pe_cbst_report_trainer import CbstPeReportTrainer
from src.specific.cbst.trainer.pe_cbst_data_trainer import CbstPeDataTrainer
from src.specific.cbst.trainer.pe_cbst_model_trainer import CbstPeModelTrainer
from src.specific.cbst.trainer.pe_cbst_logic_trainer import CbstPeLogicTrainer
from src.specific.cbst.trainer.pe_cbst_train_algo_args import CbstPeTrainAlgoArgs
from src.specific.cbst.trainer.pe_cbst_train_report_args import CbstPeTrainReportArgs
from src.specific.cbst.trainer.pe_cbst_io_trainer import CbstPeIoTrainer
from src.specific.cbst.trainer.pe_cbst_train_writer import CbstPeTrainWriter
from src.specific.cbst.trainer.pe_cbst_output_mapper import CbstPeOutputMapper
from src.specific.cbst.trainer.pe_cbst_output_writer import CbstPeOutputWriter


class CbstPeTrainerProvider:
    """Dependency injection factory for CatBoost trainer components."""

    @staticmethod
    def get_io_trainer():
        """Create fully configured I/O trainer with all dependencies."""
        model_trainer = CbstPeModelTrainer()
        data_trainer = CbstPeDataTrainer()
        report_trainer = CbstPeReportTrainer(data_trainer, model_trainer)
        logic_trainer = CbstPeLogicTrainer(data_trainer, model_trainer, report_trainer)
        cbst_output_mapper = CbstPeOutputMapper()
        output_writer = CbstPeOutputWriter()
        cbst_writer = CbstPeTrainWriter(cbst_output_mapper, output_writer)
        return CbstPeIoTrainer(data_trainer, logic_trainer, cbst_writer)
