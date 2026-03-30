from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate


class LrPeModelTrainer:

    def get_logistic_regression_classifier(self, args):
        return LogisticRegression(
            solver=args.solver,
            max_iter=args.max_iter,
            random_state=args.random_state,
            class_weight='balanced'
        )

    def get_split_train_test(self, args):
        return StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    def get_cross_validate_score(self, model, skf, ml_features, ml_label):
        return cross_validate(model, ml_features, ml_label, cv=skf, scoring=['roc_auc', 'accuracy'], n_jobs=-1)

    def fit_model(self, model, ml_features, ml_label):
        return model.fit(ml_features, ml_label)

    def build(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)
