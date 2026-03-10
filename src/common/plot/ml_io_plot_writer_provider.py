from src.common.image.file_io_validator import FileIoValidator
from src.common.plot import MatplotlibPlotRenderer, MatplotlibPlotExporter, MlIoPlotWriter
from src.common.plot.confusion_matrix_spec_factory import ConfusionMatrixPlotSpecFactory


class MlIoPlotWriterProvider:

    @staticmethod
    def get_writer():
        file_validator = FileIoValidator()
        spec_factory = ConfusionMatrixPlotSpecFactory()
        plot_renderer = MatplotlibPlotRenderer()
        plot_exporter = MatplotlibPlotExporter(plot_renderer)
        plot_writer = MlIoPlotWriter(file_validator, spec_factory, plot_exporter)
        return plot_writer

