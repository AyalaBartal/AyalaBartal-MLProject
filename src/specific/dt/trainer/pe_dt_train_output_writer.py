import json

import joblib
import pandas as pd
from sklearn.tree import export_graphviz


class DtPeTrainOutputWriter:

    def write_text_to_text_file(self, out_file, end_message):
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(end_message)

    def write_model_to_joblib_file(self, out_file, dt_model):
        joblib.dump(dt_model, out_file)

    def write_object_to_json_file(self, out_put, data):
        with open(out_put, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def write_list_to_json_file(self, out_file, data):
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump({'feature_order': list(data)}, f)


    def write_dt_model_to_graphviz_dot_file(self, out_file, feature_names, dt_model):
        export_graphviz(
            dt_model,
            out_file=str(out_file),
            feature_names=feature_names,
            class_names=[str(c) for c in dt_model.classes_],
            filled=True,
            rounded=True
        )

    def write_feature_importance_to_csv(self, out_file, feature_names, dt_model):
        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": dt_model.feature_importances_
        }).sort_values(by="importance", ascending=False)
        feature_importance.to_csv(out_file, index=False)
