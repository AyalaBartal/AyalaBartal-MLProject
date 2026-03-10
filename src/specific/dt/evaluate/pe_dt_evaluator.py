import pandas as pd
import json, os
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from joblib import load

from src.common.image.file_io_validator import FileIoValidator
from src.common.plot import MlIoPlotWriter
from src.common.plot.confusion_matrix_spec_factory import ConfusionMatrixPlotSpecFactory
from src.common.plot.matplotlib_plot_renderer import MatplotlibPlotRenderer
from src.specific.dt.evaluate.file_util import FileUtil


class DtPeDataEvaluator:

    def evaluate(self, args):
        print("Start DtPeDataEvaluator")
        os.makedirs(args.output_dir, exist_ok=True)
        FileUtil.is_readable_directory(args.input_dir, True, False)
        FileUtil.is_readable_directory(args.output_dir, True, True)

        df = pd.read_csv(args.input_csv)
        model = load(args.input_model)

        column_label = args.column_label
        y = df[column_label].values
        x = df.drop(columns=[column_label])

        proba = DtPeDataEvaluator.get_proba(model, x)
        y_pred = (proba >= args.threshold).astype(int)

        auc = roc_auc_score(y, proba)
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)

        self.write_out_json(acc, args, auc, cm)
        self.write_out_md(acc, args, auc, cm)

        file_validator = FileIoValidator()
        spec_factory = ConfusionMatrixPlotSpecFactory()
        plot_renderer = MatplotlibPlotRenderer()
        plot_writer = MlIoPlotWriter(file_validator, spec_factory, plot_renderer)
        plot_writer.create_plot(args.out_png, cm)

        print("End DtPeDataEvaluator")

    @staticmethod
    def get_proba(model, x):
        is_proba = hasattr(model, 'predict_proba')
        if is_proba:
            predict_proba = model.predict_proba(x)
            return predict_proba[:, 1]
        return (lambda d: (d - d.min()) / (d.max() - d.min() + 1e-9))(model.decision_function(x))


    def write_out_json(self, acc, args, auc, cm):
        json_data = DtPeDataEvaluator.build_accuracy_measure(acc, args, auc, cm)
        json.dump(json_data, open(args.out_json, 'w'), indent=2)

    def write_out_md(self, acc, args, auc, cm):
        md = f"""# Decision Tree — Test Metrics\n\n- AUC: {auc:.4f}\n- Accuracy: {acc:.4f}\n- Threshold: {args.threshold:.2f}\n\n**Confusion Matrix** (rows=Actual, cols=Predicted):\n{cm.tolist()}\n"""
        open(args.out_md, 'w').write(md)
        print(md)


    @staticmethod
    def build_accuracy_measure(acc, args, auc, cm):
        return {'auc': float(auc),
                'accuracy': float(acc),
                'threshold': args.threshold,
                'confusion_matrix': cm.tolist()}



