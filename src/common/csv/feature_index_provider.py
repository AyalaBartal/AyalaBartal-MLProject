class CsvBrazilianProvider:

    @staticmethod
    def get_map_header_by_index():
        column_header_by_indexes = {
            0: "BaseOfCode",
            1: "BaseOfData",
            2: "Characteristics",
            3: "DllCharacteristics",
            4: "Entropy",
            5: "FileAlignment",
            6: "FirstSeenDate",
            7: "Identify",
            8: "ImageBase",
            9: "ImportedDlls",
            10: "ImportedSymbols",
            11: "Label",
            12: "Machine",
            13: "Magic",
            14: "NumberOfRvaAndSizes",
            15: "NumberOfSections",
            16: "NumberOfSymbols",
            17: "PE_TYPE",
            18: "PointerToSymbolTable",
            19: "SHA1",
            20: "Size",
            21: "SizeOfCode",
            22: "SizeOfHeaders",
            23: "SizeOfImage",
            24: "SizeOfInitializedData",
            25: "SizeOfOptionalHeader",
            26: "SizeOfUninitializedData",
            27: "TimeDateStamp"
        }
        return column_header_by_indexes

    @staticmethod
    def get_non_negative_integer_headers():
        return [
            "BaseOfCode",
            "BaseOfData",
            "FileAlignment",
            "ImageBase",
            "Machine",
            "Magic",
            "NumberOfRvaAndSizes",
            "NumberOfSections",
            "NumberOfSymbols",
            "PE_TYPE",
            "PointerToSymbolTable",
            "Size",
            "SizeOfCode",
            "SizeOfHeaders",
            "SizeOfImage",
            "SizeOfInitializedData",
            "SizeOfOptionalHeader",
            "SizeOfUninitializedData",
            "TimeDateStamp"
        ]

    @staticmethod
    def get_flags_headers():
        return [
            "Characteristics",
            "DllCharacteristics",
        ]

    @staticmethod
    def get_positive_float_headers():
        return [
            "Entropy"
        ]

    @staticmethod
    def get_date_headers():
        return [
            "FirstSeenDate"
        ]

    @staticmethod
    def get_text_headers():
        return [
            "Identify",
            "ImportedDlls",
            "ImportedSymbols",
        ]

    @staticmethod
    def get_hash_headers():
        return [
            "SHA1"
        ]

    @staticmethod
    def get_binary_headers():
        return [
            "Label"
        ]



