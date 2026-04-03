from src.common.validator.args_validator import ArgsValidator
from src.specific.xgb.trainer.pe_xgb_train_result import XgbPeTrainResult
from src.specific.xgb.trainer.pe_xgb_model_trainer import XgbPeModelTrainer
from src.specific.xgb.trainer.pe_xgb_data_trainer import XgbPeDataTrainer


class XgbPeReportTrainer:
    """Report generation and confusion matrix calculation."""

    def __init__(self, row_selector: XgbPeDataTrainer, matrix_builder: XgbPeModelTrainer):
        ArgsValidator.require_type_not_none(row_selector, XgbPeDataTrainer, "row_selector")
        ArgsValidator.require_type_not_none(matrix_builder, XgbPeModelTrainer, "matrix_builder")
        self.row_selector = row_selector
        self.matrix_builder = matrix_builder

    def get_confusion_matrix(self, model, cv, x, y):
        """
        Calculate confusion matrix by:
        - iterating CV folds
        - selecting train/test rows
        - fitting and predicting
        - delegating final matrix creation
        """
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
        """Generate training result object."""
        return XgbPeTrainResult(args, ml_features, model, con_matrix, scores['test_accuracy'], scores['test_roc_auc'])
