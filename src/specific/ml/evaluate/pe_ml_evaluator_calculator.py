import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from src.specific.ml.evaluate.pe_ml_evaluate_report import MlPeEvaluateReport


class MlPeEvaluatorCalculator:

    @staticmethod
    def get_input_from_data_label(args_al, df):
        return df.drop(columns=[args_al.column_label])

    @staticmethod
    def get_output_from_data_label(args_al, df):
        return df[args_al.column_label].values

    @staticmethod
    def get_prob_from_model_x(model, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        x_tensor = torch.FloatTensor(x.values).to(device)
        with torch.no_grad():
            output = model(x_tensor)
        
        if output.shape[1] == 2:
            proba = torch.softmax(output, dim=1)
            return proba[:, 1].cpu().numpy()
        else:
            return torch.sigmoid(output).squeeze().cpu().numpy()

    @staticmethod
    def get_pred_from_prob_threshold(proba, threshold):
        return (proba >= threshold).astype(int)

    @staticmethod
    def get_report_from_y_prob_pred(y, prob, pred):
        auc = roc_auc_score(y, prob)
        acc = accuracy_score(y, pred)
        cm = confusion_matrix(y, pred)
        return MlPeEvaluateReport(auc, acc, cm)
