from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate


class DtPeModelTrainer:

    # Build a decision tree classifier. It contains both model and algorithm to train it. OO design.
    def get_decision_tree_classifier(self, args):
        return DecisionTreeClassifier(criterion=args.criterion,
                                      max_depth=args.max_depth,
                                      min_samples_leaf=args.min_samples_leaf,
                                      class_weight='balanced',
                                      random_state=args.random_state)

    # Provides train/test indices to split data in train/test sets
    def get_split_train_test(self, args):
        return StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    def get_cross_validate_score(self, model, skf, ml_features, ml_label):
        return cross_validate(model, ml_features, ml_label, cv=skf, scoring=['roc_auc', 'accuracy'], n_jobs=-1)

    # Build a decision tree classifier from the training set (ml_features, ml_label).
    def fit_model(self, model, ml_features, ml_label):
        return model.fit(ml_features, ml_label)

    # Build confusion matrix using sklearn.metrics.confusion_matrix.
    def build(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)
