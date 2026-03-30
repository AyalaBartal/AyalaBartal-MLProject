import os
import json
from joblib import dump


class LrPeTrainOutputWriter:

    def write_model(self, output_dir, model):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'logistic_regression_model.joblib')
        dump(model, model_path)
        return model_path

    def write_metrics(self, output_dir, metrics):
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, 'lr_cv_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        return metrics_path

    def write_feature_schema(self, output_dir, feature_schema):
        os.makedirs(output_dir, exist_ok=True)
        schema_path = os.path.join(output_dir, 'lr_feature_schema.json')
        with open(schema_path, 'w') as f:
            json.dump(feature_schema, f, indent=2)
        return schema_path
