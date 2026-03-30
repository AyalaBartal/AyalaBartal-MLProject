import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score


class MLPModel(nn.Module):
    """PyTorch MLP model for binary classification"""

    def __init__(self, input_size, hidden_sizes, num_classes=2):
        super(MLPModel, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MlPeModelTrainer:

    def get_mlp_model(self, input_size, args):
        return MLPModel(input_size, args.hidden_sizes, num_classes=2)

    def get_split_train_test(self, args):
        return StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    def fit_model(self, model, X_train, y_train, args):
        """Train MLP model using DataLoader"""
        X_train_tensor = torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train)
        y_train_tensor = torch.LongTensor(y_train.values if hasattr(y_train, 'values') else y_train)
        
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        model.train()
        for epoch in range(args.epochs):
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        
        return model

    def predict(self, model, X_test):
        """Generate predictions from trained model"""
        X_test_tensor = torch.FloatTensor(X_test.values if hasattr(X_test, 'values') else X_test)
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predictions = torch.max(outputs, 1)
        return predictions.numpy()

    def predict_proba(self, model, X_test):
        """Generate probability predictions"""
        X_test_tensor = torch.FloatTensor(X_test.values if hasattr(X_test, 'values') else X_test)
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.numpy()

    def build(self, y_true, y_pred):
        return self._build_confusion_matrix(y_true, y_pred)

    def _build_confusion_matrix(self, y_true, y_pred):
        y_true_np = y_true.values if hasattr(y_true, 'values') else y_true
        y_pred_np = y_pred if isinstance(y_pred, np.ndarray) else np.array(y_pred)
        
        return {
            'true_neg': int(((y_true_np == 0) & (y_pred_np == 0)).sum()),
            'false_pos': int(((y_true_np == 0) & (y_pred_np == 1)).sum()),
            'false_neg': int(((y_true_np == 1) & (y_pred_np == 0)).sum()),
            'true_pos': int(((y_true_np == 1) & (y_pred_np == 1)).sum())
        }
