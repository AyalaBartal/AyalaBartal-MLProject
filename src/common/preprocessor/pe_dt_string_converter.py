import json
import math
import re


class DtPeStringConverter:

    # Convert a single  value (number, json, sentence) into a list of simple words.
    @staticmethod
    def parse_listish(value):
        text = str(value).strip()
        # Case 1: value is empty: None and NaN. NaN is float.
        if value is None or text.lower() in {"nan", "none", ""}:
            return []
        # Case 2: value is an empty json.
        if text == "" or text == "[]" or text == "{}":
            return []
        # Case 3: value is a non-empty json: ["a","b"] or [1,2,3], [{"a":1}, {"b":2}]
        if text.startswith("[") and text.endswith("]"):
            json_data = json.loads(text)
            return [str(item) for item in json_data if item is not None]
        # Case 4: default fallback, split by common separators and filter out empty words
        words = re.split(r"[\s,;|]+", text)
        return [w for w in words if w]

    # Normalizes DLL name from path and .dll extension.
    @staticmethod
    def clean_dll(value):
        # Convert into String, lower case,remove surrounding whitespace and quotes.
        value = str(value).lower().strip().strip('"\'')
        # Remove file path (Windows and Linux)
        value = value.split('\\')[-1].split('/')[-1]
        # Remove suffix .dll, if exists, by removing the last 4 chars.
        if value.endswith('.dll'):
            value = value[:-4]
        return value

    # Normalizes API/method path. Example: [ "kernel32!CreateFileW" ] --> [createfilew]
    @staticmethod
    def clean_api(value):
        # Convert into String, lower case,remove surrounding whitespace and quotes.
        value = str(value).lower().strip().strip('"\'')
        # Extracts from value only the function name: kernel32!createfilew --> createfilew
        func = value.split('!', 1)[1] if '!' in value else value
        # Replace special characters with space: create-file@1 --> create file
        func = re.sub(r"[^a-z0-9_]+", " ", func)
        # Split sentence by space into words: "create file w" -> ["create", "file", "w"]
        # Filter out short words with less than three chars. Add only and all long words to output.
        return [w for w in func.split() if len(w) >= 2]

    # Normalizes an identifier string and converts it into words
    @staticmethod
    def clean_ident(value):
        # If value is empty: None and NaN. NaN is float. Then return empty array.
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return []
        # Convert into String, lower case, Replace non-identifier characters with space.
        s = re.sub(r"[^a-z0-9_]+", " ", str(value).lower())
        # Split sentence by space into words: "create file w" -> ["create", "file", "w"]
        # Filter out short words with less than three chars. Add only and all long words to output. 1
        return [w for w in s.split() if len(w) >= 2]