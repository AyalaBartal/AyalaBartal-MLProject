import unittest
from unittest.mock import Mock, call

from src.common.plot import MlIoPlotWriter


class TestMlIoPlotWriter(unittest.TestCase):

    def setUp(self):
        self.validator = Mock()
        self.factory = Mock()
        self.exporter = Mock()

        self.writer = MlIoPlotWriter(
            validator=self.validator,
            factory=self.factory,
            exporter=self.exporter,
        )

    def test_create_plot_calls_dependencies_in_order(self):
        out_png = "out/confusion_matrix.png"
        cm = [[12, 3], [2, 9]]
        spec = Mock(name="plot_spec")

        self.factory.build.return_value = spec

        self.writer.create_plot(out_png, cm)

        self.validator.validate_file_writeable.assert_called_once_with(out_png)
        self.factory.validate_confusion_matrix_data.assert_called_once_with(cm)
        self.factory.build.assert_called_once_with(cm)
        self.exporter.render_and_save.assert_called_once_with(spec, out_png)

        self.assertEqual(
            [
                call.validate_file_writeable(out_png),
                call.validate_confusion_matrix_data(cm),
                call.build(cm),
            ],
            self.validator.mock_calls + self.factory.mock_calls,
        )

    def test_create_plot_passes_built_spec_to_renderer(self):
        out_png = "plot.png"
        cm = [[1, 2], [3, 4]]
        spec = object()

        self.factory.build.return_value = spec

        self.writer.create_plot(out_png, cm)

        args, kwargs = self.exporter.render_and_save.call_args
        self.assertIs(args[0], spec)
        self.assertEqual(args[1], out_png)
        self.assertEqual(kwargs, {})

    def test_create_plot_stops_when_output_file_is_not_writeable(self):
        out_png = "plot.png"
        cm = [[1, 0], [0, 1]]

        self.validator.validate_file_writeable.side_effect = PermissionError(
            "cannot write output file"
        )

        with self.assertRaises(PermissionError):
            self.writer.create_plot(out_png, cm)

        self.factory.validate_confusion_matrix_data.assert_not_called()
        self.factory.build.assert_not_called()
        self.exporter.render_and_save.assert_not_called()

    def test_create_plot_stops_when_confusion_matrix_is_invalid(self):
        out_png = "plot.png"
        cm = [[1, 2, 3], [4, 5, 6]]

        self.factory.validate_confusion_matrix_data.side_effect = ValueError(
            "cm must be a 2x2 confusion matrix"
        )

        with self.assertRaises(ValueError):
            self.writer.create_plot(out_png, cm)

        self.validator.validate_file_writeable.assert_called_once_with(out_png)
        self.factory.validate_confusion_matrix_data.assert_called_once_with(cm)
        self.factory.build.assert_not_called()
        self.exporter.render_and_save.assert_not_called()

    def test_create_plot_does_not_render_when_build_fails(self):
        out_png = "plot.png"
        cm = [[1, 2], [3, 4]]

        self.factory.build.side_effect = RuntimeError("failed to build spec")

        with self.assertRaises(RuntimeError):
            self.writer.create_plot(out_png, cm)

        self.validator.validate_file_writeable.assert_called_once_with(out_png)
        self.factory.validate_confusion_matrix_data.assert_called_once_with(cm)
        self.factory.build.assert_called_once_with(cm)
        self.exporter.render_and_save.assert_not_called()
