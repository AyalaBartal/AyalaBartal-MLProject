class MlIoPlotWriter:

    def __init__(self, validator, factory, exporter):
        self.file_validator = validator
        self.spec_factory = factory
        self.plot_exporter = exporter

    def create_plot(self, out_png, cm):
        self.file_validator.validate_file_writeable(out_png)
        self.spec_factory.validate_confusion_matrix_data(cm)
        spec = self.spec_factory.build(cm)
        self.plot_exporter.render_and_save(spec, out_png)
