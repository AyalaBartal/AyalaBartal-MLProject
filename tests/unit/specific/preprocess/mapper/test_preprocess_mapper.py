import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.specific.dt.preprocess.pe_dt_preprocess_mapper import DtPePreprocessMapper
from src.specific.dt.preprocess.pe_dt_preprocess_map_args import DtPeDataPreprocessMapArgs


class TestDtPePreprocessMapper(unittest.TestCase):

    def test_map_empty_transform_output_returns_empty_dataframe_with_same_index(self):
        args = DtPeDataPreprocessMapArgs()
        transformer = Mock()
        transformer.transform.return_value = []

        mapper = DtPePreprocessMapper(args, transformer)

        input_data = pd.DataFrame(
            {
                "A": [1, 2],
                "B": [3, 4],
            },
            index=[10, 11]
        )

        actual = mapper.map(input_data)

        expected = pd.DataFrame(columns=[], index=input_data.index)
        assert_frame_equal(actual, expected)
        transformer.transform.assert_called_once_with(input_data)

    def test_map_concats_transformer_outputs_by_columns(self):
        args = DtPeDataPreprocessMapArgs()
        transformer = Mock()

        df1 = pd.DataFrame({"f1": [1, 2]}, index=[0, 1])
        df2 = pd.DataFrame({"f2": [3, 4]}, index=[0, 1])
        transformer.transform.return_value = [df1, df2]

        mapper = DtPePreprocessMapper(args, transformer)

        input_data = pd.DataFrame({"raw": ["a", "b"]})

        actual = mapper.map(input_data)

        expected = pd.DataFrame(
            {
                "f1": [1, 2],
                "f2": [3, 4],
            },
            index=[0, 1]
        ).sort_index(axis=1)

        assert_frame_equal(actual, expected)

    def test_map_replaces_inf_nan_and_minus_inf_with_zero(self):
        args = DtPeDataPreprocessMapArgs()
        transformer = Mock()

        df1 = pd.DataFrame(
            {
                "a": [1.0, np.inf, np.nan],
                "b": [-np.inf, 5.0, 6.0],
            }
        )
        transformer.transform.return_value = [df1]

        mapper = DtPePreprocessMapper(args, transformer)

        input_data = pd.DataFrame({"raw": [10, 20, 30]})

        actual = mapper.map(input_data)

        expected = pd.DataFrame(
            {
                "a": [1.0, 0.0, 0.0],
                "b": [0.0, 5.0, 6.0],
            }
        ).sort_index(axis=1)

        assert_frame_equal(actual, expected)

    def test_map_sanitizes_column_names(self):
        args = DtPeDataPreprocessMapArgs()
        transformer = Mock()

        df1 = pd.DataFrame(
            {
                "api.create-file": [1, 2],
                "dll user32!": [3, 4],
                "a/b\\c": [5, 6],
            }
        )
        transformer.transform.return_value = [df1]

        mapper = DtPePreprocessMapper(args, transformer)

        input_data = pd.DataFrame({"raw": [10, 20]})

        actual = mapper.map(input_data)

        expected = pd.DataFrame(
            {
                "api_create_file": [1, 2],
                "dll_user32_": [3, 4],
                "a_b_c": [5, 6],
            }
        ).sort_index(axis=1)

        assert_frame_equal(actual, expected)

    def test_map_adds_label_column_when_present_in_input(self):
        args = DtPeDataPreprocessMapArgs()
        args.label_col = "Label"

        transformer = Mock()
        df1 = pd.DataFrame({"f1": [10, 20]})
        transformer.transform.return_value = [df1]

        mapper = DtPePreprocessMapper(args, transformer)

        input_data = pd.DataFrame(
            {
                "raw": ["x", "y"],
                "Label": [1, 0],
            }
        )

        actual = mapper.map(input_data)

        expected = pd.DataFrame(
            {
                "Label": [1, 0],
                "f1": [10, 20],
            }
        ).sort_index(axis=1)

        assert_frame_equal(actual, expected)

    def test_map_does_not_add_label_column_when_missing_in_input(self):
        args = DtPeDataPreprocessMapArgs()
        args.label_col = "Label"

        transformer = Mock()
        df1 = pd.DataFrame({"f1": [10, 20]})
        transformer.transform.return_value = [df1]

        mapper = DtPePreprocessMapper(args, transformer)

        input_data = pd.DataFrame({"raw": ["x", "y"]})

        actual = mapper.map(input_data)

        expected = pd.DataFrame({"f1": [10, 20]}).sort_index(axis=1)

        assert_frame_equal(actual, expected)

    def test_map_does_not_add_label_when_label_col_is_empty(self):
        args = DtPeDataPreprocessMapArgs()
        args.label_col = ""

        transformer = Mock()
        df1 = pd.DataFrame({"f1": [10, 20]})
        transformer.transform.return_value = [df1]

        mapper = DtPePreprocessMapper(args, transformer)

        input_data = pd.DataFrame(
            {
                "raw": ["x", "y"],
                "Label": [1, 0],
            }
        )

        actual = mapper.map(input_data)

        expected = pd.DataFrame({"f1": [10, 20]}).sort_index(axis=1)

        assert_frame_equal(actual, expected)

    def test_map_sorts_columns_alphabetically(self):
        args = DtPeDataPreprocessMapArgs()
        args.label_col = "Label"

        transformer = Mock()
        df1 = pd.DataFrame({"z_col": [1, 2], "a_col": [3, 4]})
        transformer.transform.return_value = [df1]

        mapper = DtPePreprocessMapper(args, transformer)

        input_data = pd.DataFrame({"Label": [0, 1]})

        actual = mapper.map(input_data)

        self.assertEqual(list(actual.columns), ["Label", "a_col", "z_col"])

    def test_map_empty_transform_output_with_label_column_returns_only_label(self):
        args = DtPeDataPreprocessMapArgs()
        args.label_col = "Label"

        transformer = Mock()
        transformer.transform.return_value = []

        mapper = DtPePreprocessMapper(args, transformer)

        input_data = pd.DataFrame(
            {
                "raw": ["x", "y"],
                "Label": [1, 0],
            },
            index=[5, 6]
        )

        actual = mapper.map(input_data)

        expected = pd.DataFrame(
            {
                "Label": [1, 0],
            },
            index=[5, 6]
        ).sort_index(axis=1)

        assert_frame_equal(actual, expected)