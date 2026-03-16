import unittest

from src.specific.dt.preprocess import DtPeStringConverter


class TestDtPeStringConverter(unittest.TestCase):

    # ---------- parse_listish ----------

    def test_parse_listish_none_returns_empty_list(self):
        self.assertEqual(DtPeStringConverter.parse_listish(None), [])

    def test_parse_listish_nan_string_returns_empty_list(self):
        self.assertEqual(DtPeStringConverter.parse_listish(float("nan")), [])

    def test_parse_listish_empty_values_return_empty_list(self):
        self.assertEqual(DtPeStringConverter.parse_listish(""), [])
        self.assertEqual(DtPeStringConverter.parse_listish("   "), [])
        self.assertEqual(DtPeStringConverter.parse_listish("none"), [])
        self.assertEqual(DtPeStringConverter.parse_listish("nan"), [])

    def test_parse_listish_empty_json_return_empty_list(self):
        self.assertEqual(DtPeStringConverter.parse_listish("[]"), [])
        self.assertEqual(DtPeStringConverter.parse_listish("{}"), [])

    def test_parse_listish_json_list_of_strings(self):
        actual = DtPeStringConverter.parse_listish('["a", "b", "c"]')
        self.assertEqual(actual, ["a", "b", "c"])

    def test_parse_listish_json_list_of_numbers(self):
        actual = DtPeStringConverter.parse_listish("[1, 2, 3]")
        self.assertEqual(actual, ["1", "2", "3"])

    def test_parse_listish_json_list_skips_none(self):
        actual = DtPeStringConverter.parse_listish('["a", null, "b"]')
        self.assertEqual(actual, ["a", "b"])

    def test_parse_listish_json_list_of_objects(self):
        actual = DtPeStringConverter.parse_listish('[{"a": 1}, {"b": 2}]')
        self.assertEqual(actual, ["{'a': 1}", "{'b': 2}"])

    def test_parse_listish_fallback_split_by_common_separators(self):
        actual = DtPeStringConverter.parse_listish("a, b; c|d e")
        self.assertEqual(actual, ["a", "b", "c", "d", "e"])

    # ---------- clean_dll ----------

    def test_clean_dll_removes_windows_path_and_extension(self):
        actual = DtPeStringConverter.clean_dll(r'C:\Windows\System32\KERNEL32.DLL')
        self.assertEqual(actual, "kernel32")

    def test_clean_dll_removes_linux_path_and_extension(self):
        actual = DtPeStringConverter.clean_dll('/usr/lib/libssl.dll')
        self.assertEqual(actual, "libssl")

    def test_clean_dll_removes_quotes_and_spaces(self):
        actual = DtPeStringConverter.clean_dll('  "User32.dll"  ')
        self.assertEqual(actual, "user32")

    def test_clean_dll_keeps_value_when_no_extension(self):
        actual = DtPeStringConverter.clean_dll("ntdll")
        self.assertEqual(actual, "ntdll")

    # ---------- clean_api ----------

    def test_clean_api_extracts_function_name_after_bang(self):
        actual = DtPeStringConverter.clean_api("kernel32!CreateFileW")
        self.assertEqual(actual, ["createfilew"])

    def test_clean_api_replaces_special_chars_with_spaces(self):
        actual = DtPeStringConverter.clean_api("create-file@1")
        self.assertEqual(actual, ["create", "file"])

    def test_clean_api_removes_short_tokens(self):
        actual = DtPeStringConverter.clean_api("a bb ccc d")
        self.assertEqual(actual, ["bb", "ccc"])

    def test_clean_api_removes_quotes_and_lowercases(self):
        actual = DtPeStringConverter.clean_api('  "NTDLL!ZwOpenFile" ')
        self.assertEqual(actual, ["zwopenfile"])

    # ---------- clean_ident ----------

    def test_clean_ident_none_returns_empty_list(self):
        self.assertEqual(DtPeStringConverter.clean_ident(None), [])

    def test_clean_ident_nan_returns_empty_list(self):
        self.assertEqual(DtPeStringConverter.clean_ident(float("nan")), [])

    def test_clean_ident_normalizes_and_splits_words(self):
        actual = DtPeStringConverter.clean_ident("Create-File_Name")
        self.assertEqual(actual, ["create", "file_name"])

    def test_clean_ident_removes_short_tokens(self):
        actual = DtPeStringConverter.clean_ident("a bb ccc d")
        self.assertEqual(actual, ["bb", "ccc"])

    def test_clean_ident_handles_numbers_and_symbols(self):
        actual = DtPeStringConverter.clean_ident("API_v2@Build#45")
        self.assertEqual(actual, ["api_v2", "build", "45"])
