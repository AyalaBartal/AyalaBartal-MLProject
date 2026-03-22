import unittest
from unittest.mock import patch, mock_open

from src.specific.dt.evaluate import FileWriter


class TestDtFileWriter(unittest.TestCase):

    def test_write_out_json_writes_expected_content(self):
        out_path = "test.json"
        data = {"a": 1, "b": 2}

        m = mock_open()

        with patch("builtins.open", m), patch("json.dump") as mock_json_dump:
            FileWriter.write_out_json(out_path, data)

            # Verify file opened correctly
            m.assert_called_once_with(out_path, 'w')

            # Verify json.dump called with correct args
            mock_json_dump.assert_called_once()
            args, kwargs = mock_json_dump.call_args

            self.assertEqual(args[0], data)        # json_data
            self.assertEqual(kwargs["indent"], 2)  # indent
            # args[1] is the file handle → no need to assert exact object

    def test_write_out_md_writes_expected_text(self):
        out_path = "test.md"
        text = "# Hello World"

        m = mock_open()

        with patch("builtins.open", m):
            FileWriter.write_out_md(out_path, text)

            # Verify file opened correctly
            m.assert_called_once_with(out_path, 'w')

            # Verify write called with correct text
            handle = m()
            handle.write.assert_called_once_with(text)
