import unittest

from src.specific.dt.preprocess import DtPeListConverter


class TestDtPeListConverter(unittest.TestCase):

    # ---------- clean_dlls ----------

    def test_clean_dlls_empty_input_returns_empty_list(self):
        self.assertEqual(DtPeListConverter.clean_dlls([]), [])

    def test_clean_dlls_single_dll(self):
        actual = DtPeListConverter.clean_dlls([r"C:\Windows\System32\KERNEL32.DLL"])
        self.assertEqual(actual, list("kernel32"))

    def test_clean_dlls_multiple_dlls(self):
        actual = DtPeListConverter.clean_dlls([
            r"C:\Windows\System32\KERNEL32.DLL",
            "/usr/lib/libssl.dll",
            '  "User32.dll"  ',
        ])
        self.assertEqual(
            actual,
            list("kernel32") + list("libssl") + list("user32")
        )

    def test_clean_dlls_value_without_extension(self):
        actual = DtPeListConverter.clean_dlls(["ntdll"])
        self.assertEqual(actual, list("ntdll"))

    # ---------- clean_apis ----------

    def test_clean_apis_empty_input_returns_empty_list(self):
        self.assertEqual(DtPeListConverter.clean_apis([]), [])

    def test_clean_apis_single_api(self):
        actual = DtPeListConverter.clean_apis(["kernel32!CreateFileW"])
        self.assertEqual(actual, ["createfilew"])

    def test_clean_apis_multiple_apis(self):
        actual = DtPeListConverter.clean_apis([
            "kernel32!CreateFileW",
            "ntdll!ZwOpenFile",
            "create-file@1",
        ])
        self.assertEqual(
            actual,
            ["createfilew", "zwopenfile", "create", "file"]
        )

    def test_clean_apis_skips_short_tokens(self):
        actual = DtPeListConverter.clean_apis(["a bb ccc d"])
        self.assertEqual(actual, ["bb", "ccc"])
