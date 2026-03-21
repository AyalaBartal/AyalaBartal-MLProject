import json
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from src.common.image.file_io_validator import FileIoValidator
from src.common.plot import MlIoPlotWriter, MatplotlibPlotExporter
from src.common.plot.confusion_matrix_spec_factory import ConfusionMatrixPlotSpecFactory
from src.common.plot.matplotlib_plot_renderer import MatplotlibPlotRenderer
from src.specific.dt.evaluate import DtPeEvaluateInputArgs, DtPeEvaluateOutputArgs, DtPeEvaluateAlgoArgs, \
    DtPeEvaluateReport
from src.specific.dt.evaluate.pe_dt_evaluator_reader import DtPeEvaluatorReader


class DtPeDataEvaluator:

    def __init__(self, reader: DtPeEvaluatorReader):
        self.reader = reader

    def evaluate(self, args_in: DtPeEvaluateInputArgs, args_al: DtPeEvaluateAlgoArgs, args_out: DtPeEvaluateOutputArgs):

        print("Start DtPeDataEvaluator")
        self.reader.validate_input(args_in)
        self.reader.validate_output(args_out)

        df = self.reader.read_csv_to_df(args_in.input_csv)
        model = self.reader.read_dt_model_from_joblib_file(args_in)

        y = df[args_al.column_label].values
        x = df.drop(columns=[args_al.column_label])

        proba = DtPeDataEvaluator.get_proba(model, x)
        y_pred = self.calc_pred(args_al, proba)
        report = self.calculate_y_prob_pred(y, proba, y_pred)

        self.write_out_json(args_out.out_json, args_al.threshold, report)
        self.write_out_md(args_out.out_md, args_al.threshold, report)

        plot_writer = self.get_plot_writer()
        plot_writer.create_plot(args_out.out_png, report.cm)

        print("End DtPeDataEvaluator")



    def get_plot_writer(self):
        file_validator = FileIoValidator()
        spec_factory = ConfusionMatrixPlotSpecFactory()
        plot_renderer = MatplotlibPlotRenderer()
        plot_exporter = MatplotlibPlotExporter(plot_renderer)
        plot_writer = MlIoPlotWriter(file_validator, spec_factory, plot_exporter)
        return plot_writer

    @staticmethod
    def calc_pred(args_al, proba):
        return (proba >= args_al.threshold).astype(int)

    @staticmethod
    def calculate_y_prob_pred(y, prob, pred):
        auc = roc_auc_score(y, prob)
        acc = accuracy_score(y, pred)
        cm = confusion_matrix(y, pred)
        return DtPeEvaluateReport(auc, acc, cm)

    @staticmethod
    def get_proba(model, x):
        is_proba = hasattr(model, 'predict_proba')
        if is_proba:
            predict_proba = model.predict_proba(x)
            return predict_proba[:, 1]
        return (lambda d: (d - d.min()) / (d.max() - d.min() + 1e-9))(model.decision_function(x))


    def write_out_json(self, out_json, threshold, report):
        json_data = DtPeDataEvaluator.build_accuracy_measure(threshold, report)
        json.dump(json_data, open(out_json, 'w'), indent=2)

    def write_out_md(self, out_md, threshold, report):
        acc = report.acc
        auc = report.auc
        cm = report.cm
        md = f"""# Decision Tree — Test Metrics\n\n- AUC: {auc:.4f}\n- Accuracy: {acc:.4f}\n- Threshold: {threshold:.2f}\n\n**Confusion Matrix** (rows=Actual, cols=Predicted):\n{cm.tolist()}\n"""
        open(out_md, 'w').write(md)
        print(md)


    @staticmethod
    def build_accuracy_measure(threshold, report):
        return {'auc': float(report.auc),
                'accuracy': float(report.acc),
                'threshold': threshold,
                'confusion_matrix': report.cm.tolist()}



