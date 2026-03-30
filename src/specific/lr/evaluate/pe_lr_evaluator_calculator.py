from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from src.specific.lr.evaluate.pe_lr_evaluate_report import LrPeEvaluateReport


class LrPeEvaluatorCalculator:

    @staticmethod
    def get_input_from_data_label(args_al, df):
        return df.drop(columns=[args_al.column_label])

    @staticmethod
    def get_output_from_data_label(args_al, df):
        return df[args_al.column_label].values

    @staticmethod
    def get_prob_from_model_x(model, x):
        is_proba = hasattr(model, 'predict_proba')
        if is_proba:
            predict_proba = model.predict_proba(x)
            return predict_proba[:, 1]
        return (lambda d: (d - d.min()) / (d.max() - d.min() + 1e-9))(model.decision_function(x))

    @staticmethod
    def get_pred_from_prob_threshold(proba, threshold):
        return (proba >= threshold).astype(int)

    @staticmethod
    def get_report_from_y_prob_pred(y, prob, pred):
        auc = roc_auc_score(y, prob)
        acc = accuracy_score(y, pred)
        cm = confusion_matrix(y, pred)
        return LrPeEvaluateReport(auc, acc, cm)
