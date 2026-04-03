import json

import joblib
import pandas as pd


class XgbPeOutputWriter:
    """Write training results to files."""

    def write_text_to_text_file(self, out_file, end_message):
        """Write text message to markdown file."""
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(end_message)

    def write_model_to_joblib_file(self, out_file, xgb_model):
        """Serialize XGBoost model using joblib."""
        joblib.dump(xgb_model, out_file)

    def write_object_to_json_file(self, out_put, data):
        """Write dictionary to JSON file."""
        with open(out_put, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def write_list_to_json_file(self, out_file, data):
        """Write feature order list to JSON file."""
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump({'feature_order': list(data)}, f)

    def write_feature_importance_to_csv(self, out_file, feature_names, xgb_model):
        """Write feature importance scores to CSV file."""
        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": xgb_model.feature_importances_
        }).sort_values(by="importance", ascending=False)
        feature_importance.to_csv(out_file, index=False)
