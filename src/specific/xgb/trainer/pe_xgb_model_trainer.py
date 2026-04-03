from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate


class XgbPeModelTrainer:
    """XGBoost model definition and training utilities."""

    def get_xgboost_classifier(self, args):
        """Build an XGBoost binary classifier with specified hyperparameters."""
        return XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            random_state=args.random_state,
            n_jobs=-1,
            scale_pos_weight=args.scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    def get_split_train_test(self, args):
        """Provide train/test indices to split data in stratified manner."""
        return StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    def get_cross_validate_score(self, model, skf, ml_features, ml_label):
        """Perform cross-validation using ROC-AUC and accuracy metrics."""
        return cross_validate(model, ml_features, ml_label, cv=skf, scoring=['roc_auc', 'accuracy'], n_jobs=-1)

    def fit_model(self, model, ml_features, ml_label):
        """Train XGBoost classifier on the dataset."""
        return model.fit(ml_features, ml_label)

    def build(self, y_true, y_pred):
        """Build confusion matrix from true and predicted labels."""
        return confusion_matrix(y_true, y_pred)
