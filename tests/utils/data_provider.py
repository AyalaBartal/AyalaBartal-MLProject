from typing import List, Dict, Any


class DataProvider:

    @staticmethod
    def get_csv_small() -> List[Dict[str, Any]]:
        return [
            {"Name": "Alice", "Age": 30, "City": "Toronto"},
            {"Name": "Bob", "Age": 25, "City": "Ottawa"},
            {"Name": "Charlie", "Age": 35, "City": "Montreal"},
        ]
