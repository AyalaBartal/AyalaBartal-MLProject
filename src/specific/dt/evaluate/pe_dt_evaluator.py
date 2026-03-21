import pandas as pd
import json, os
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from joblib import load

from src.common.image.file_io_validator import FileIoValidator
from src.common.plot import MlIoPlotWriter, MatplotlibPlotExporter
from src.common.plot.confusion_matrix_spec_factory import ConfusionMatrixPlotSpecFactory
from src.common.plot.matplotlib_plot_renderer import MatplotlibPlotRenderer
from src.specific.dt.evaluate import DtPeEvaluateInputArgs, DtPeEvaluateOutputArgs, DtPeEvaluateAlgoArgs
from src.specific.dt.evaluate.file_util import FileUtil


class DtPeDataEvaluator:

    def evaluate(self, args_in: DtPeEvaluateInputArgs, args_al: DtPeEvaluateAlgoArgs, args_out: DtPeEvaluateOutputArgs):

        print("Start DtPeDataEvaluator")
        os.makedirs(args_out.output_dir, exist_ok=True)
        FileUtil.is_readable_directory(args_in.input_dir, True, False)
        FileUtil.is_readable_directory(args_out.output_dir, True, True)

        df = pd.read_csv(args_in.input_csv)
        model = load(args_in.input_model)

        column_label = args_al.column_label
        y = df[column_label].values
        x = df.drop(columns=[column_label])

        proba = DtPeDataEvaluator.get_proba(model, x)
        y_pred = (proba >= args_al.threshold).astype(int)

        auc = roc_auc_score(y, proba)
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)

        self.write_out_json(args_out.out_json, args_al.threshold, acc, auc, cm)
        self.write_out_md(args_out.out_md, args_al.threshold, acc, auc, cm)

        plot_writer = self.get_plot_writer()
        plot_writer.create_plot(args_out.out_png, cm)

        print("End DtPeDataEvaluator")

    def get_plot_writer(self):
        file_validator = FileIoValidator()
        spec_factory = ConfusionMatrixPlotSpecFactory()
        plot_renderer = MatplotlibPlotRenderer()
        plot_exporter = MatplotlibPlotExporter(plot_renderer)
        plot_writer = MlIoPlotWriter(file_validator, spec_factory, plot_exporter)
        return plot_writer

    @staticmethod
    def get_proba(model, x):
        is_proba = hasattr(model, 'predict_proba')
        if is_proba:
            predict_proba = model.predict_proba(x)
            return predict_proba[:, 1]
        return (lambda d: (d - d.min()) / (d.max() - d.min() + 1e-9))(model.decision_function(x))


    def write_out_json(self, out_json, threshold, acc, auc, cm):
        json_data = DtPeDataEvaluator.build_accuracy_measure(acc, threshold, auc, cm)
        json.dump(json_data, open(out_json, 'w'), indent=2)

    def write_out_md(self, out_md, threshold, acc, auc, cm):
        md = f"""# Decision Tree — Test Metrics\n\n- AUC: {auc:.4f}\n- Accuracy: {acc:.4f}\n- Threshold: {threshold:.2f}\n\n**Confusion Matrix** (rows=Actual, cols=Predicted):\n{cm.tolist()}\n"""
        open(out_md, 'w').write(md)
        print(md)


    @staticmethod
    def build_accuracy_measure(acc, threshold, auc, cm):
        return {'auc': float(auc),
                'accuracy': float(acc),
                'threshold': threshold,
                'confusion_matrix': cm.tolist()}



