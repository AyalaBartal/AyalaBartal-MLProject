from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np


class MlPeReportTrainer:

    def get_confusion_matrix(self, model, skf, ml_features, ml_label):
        con_matrix = []
        for train_idx, test_idx in skf.split(ml_features, ml_label):
            X_train, X_test = ml_features.iloc[train_idx], ml_features.iloc[test_idx]
            y_train, y_test = ml_label.iloc[train_idx], ml_label.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            con_matrix.append(self._build_confusion_matrix(y_test, y_pred))
        return con_matrix

    def get_report(self, args, ml_features, model, cv_results):
        test_auc = cv_results['test_auc']
        test_acc = cv_results['test_accuracy']

        cv_auc_mean = test_auc.mean()
        cv_auc_std = test_auc.std()
        cv_acc_mean = test_acc.mean()
        cv_acc_std = test_acc.std()

        con_matrix = cv_results.get('confusion_matrix', [])

        return {
            'cv_auc_mean': cv_auc_mean,
            'cv_auc_std': cv_auc_std,
            'cv_accuracy_mean': cv_acc_mean,
            'cv_accuracy_std': cv_acc_std,
            'confusion_matrix': con_matrix,
            'model': model,
            'feature_names': list(ml_features.columns)
        }

    def _build_confusion_matrix(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return {
            'true_neg': int(((y_true == 0) & (y_pred == 0)).sum()),
            'false_pos': int(((y_true == 0) & (y_pred == 1)).sum()),
            'false_neg': int(((y_true == 1) & (y_pred == 0)).sum()),
            'true_pos': int(((y_true == 1) & (y_pred == 1)).sum())
        }
