from src.common.validator.args_validator import ArgsValidator
from src.specific.dt.trainer.pe_dt_train_result import DtPeTrainResult
from src.specific.dt.trainer.pe_dt_model_trainer import DtPeModelTrainer
from src.specific.dt.trainer.pe_dt_data_trainer import DtPeDataTrainer


class DtPeReportTrainer:

    def __init__(self, row_selector: DtPeDataTrainer, matrix_builder: DtPeModelTrainer):
        ArgsValidator.require_type_not_none(row_selector, DtPeDataTrainer, "row_selector")
        ArgsValidator.require_type_not_none(matrix_builder, DtPeModelTrainer, "matrix_builder")
        self.row_selector = row_selector
        self.matrix_builder = matrix_builder

    """
        Calculate confusion matrix by:
        - iterating CV folds
        - selecting train/test rows
        - fitting and predicting
        - delegating final matrix creation
    """
    def get_confusion_matrix(self, model, cv, x, y):
        all_true = []
        all_pred = []

        for train_idx, test_idx in cv.split(x, y):
            x_train, x_test = self.row_selector.select_train_test(x, train_idx, test_idx)
            y_train, y_test = self.row_selector.select_train_test(y, train_idx, test_idx)

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            all_true.extend(y_test)
            all_pred.extend(y_pred)

        return self.matrix_builder.build(all_true, all_pred)

    def get_report(self, args, ml_features, model, con_matrix, scores):
        return DtPeTrainResult(args, ml_features, model, con_matrix, scores['test_accuracy'], scores['test_roc_auc'])
