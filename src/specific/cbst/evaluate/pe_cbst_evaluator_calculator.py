from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from src.specific.cbst.evaluate.pe_cbst_evaluate_report import CbstPeEvaluateReport


class CbstPeEvaluatorCalculator:
    """Metrics calculator for CatBoost evaluation.
    
    Computes evaluation metrics including AUC, accuracy, and confusion matrix
    from model predictions and true labels.
    """

    @staticmethod
    def get_input_from_data_label(args_al, df):
        """Extract input features by removing label column.
        
        Args:
            args_al: Algorithm arguments containing column_label.
            df: DataFrame containing features and labels.
            
        Returns:
            DataFrame with label column removed.
        """
        return df.drop(columns=[args_al.column_label])

    @staticmethod
    def get_output_from_data_label(args_al, df):
        """Extract output labels from DataFrame.
        
        Args:
            args_al: Algorithm arguments containing column_label.
            df: DataFrame containing features and labels.
            
        Returns:
            Array of labels.
        """
        return df[args_al.column_label].values

    @staticmethod
    def get_prob_from_model_x(model, x):
        """Get probability predictions from model.
        
        Attempts to use predict_proba if available; otherwise uses
        decision_function normalized to [0, 1] range.
        
        Args:
            model: Fitted CatBoost model.
            x: Input features.
            
        Returns:
            Array of probability predictions.
        """
        is_proba = hasattr(model, 'predict_proba')
        if is_proba:
            predict_proba = model.predict_proba(x)
            return predict_proba[:, 1]
        return (lambda d: (d - d.min()) / (d.max() - d.min() + 1e-9))(model.decision_function(x))

    @staticmethod
    def get_pred_from_prob_threshold(proba, threshold: float):
        """Convert probability predictions to binary predictions.
        
        Args:
            proba: Array of probability predictions.
            threshold: Threshold for binary classification.
            
        Returns:
            Array of binary predictions (0 or 1).
        """
        return (proba >= threshold).astype(int)

    @staticmethod
    def get_report_from_y_prob_pred(y, prob, pred) -> CbstPeEvaluateReport:
        """Compute evaluation metrics from predictions and labels.
        
        Args:
            y: True labels.
            prob: Probability predictions.
            pred: Binary predictions.
            
        Returns:
            CbstPeEvaluateReport with AUC, accuracy, and confusion matrix.
        """
        auc = roc_auc_score(y, prob)
        acc = accuracy_score(y, pred)
        cm = confusion_matrix(y, pred)
        return CbstPeEvaluateReport(auc, acc, cm)
