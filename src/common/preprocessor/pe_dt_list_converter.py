from src.common.preprocessor import DtPeStringConverter


class DtPeListConverter:

    # Normalizes DLL names from a sequence (ts) to a list of DLL names without paths and without the .dll extension.
    @staticmethod
    def clean_dlls(input_dlls):
        output_methods = []
        for value in input_dlls:
            output_methods.extend(DtPeStringConverter.clean_dll(value))
        return output_methods

    # Normalizes API/methods paths. Example: [ "kernel32!CreateFileW" ] --> [createfilew]
    @staticmethod
    def clean_apis(input_apis):
        output_apis = []
        for value in input_apis:
            output_apis.extend(DtPeStringConverter.clean_api(value))
        return output_apis
