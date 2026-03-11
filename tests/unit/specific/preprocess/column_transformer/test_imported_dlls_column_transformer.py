import unittest
from unittest.mock import Mock
import pandas as pd

from src.specific.dt.preprocess import ImportedDllsColumnTransformer


class TestImportedDllsColumnTransformer(unittest.TestCase):
    def test_process_one_cell_value(self):
        transformer = ImportedDllsColumnTransformer(
            parse_listish=Mock(return_value=["KERNEL32.dll"]),
            clean_dll=Mock(return_value=["kernel32"]),
            topk=Mock(),
            k_dlls=5,
        )

        result = transformer.process_one_cell_value("KERNEL32.dll")

        self.assertEqual(["kernel32"], result)

    def test_valid_transform(self):
        topk = Mock(return_value=pd.DataFrame({"dll_kernel32": [1, 0]}))
        transformer = ImportedDllsColumnTransformer(
            parse_listish=Mock(),
            clean_dll=Mock(),
            topk=topk,
            k_dlls=5,
        )
        data = pd.DataFrame({"ImportedDlls": ["a", "b"]})

        result = transformer.valid_transform(data, "ImportedDlls")

        self.assertEqual(1, len(result))
        pd.testing.assert_frame_equal(pd.DataFrame({"dll_kernel32": [1, 0]}), result[0])
        topk.assert_called_once_with(data["ImportedDlls"], transformer.process_one_cell_value, 5, "dll")
