import json


class FileWriter:
    """File writer for XGBoost evaluation results.
    
    Handles writing evaluation metrics to JSON and Markdown files.
    """

    @staticmethod
    def write_out_json(out_json: str, json_data: dict):
        """Write evaluation metrics to JSON file.
        
        Args:
            out_json: Path to output JSON file.
            json_data: Dictionary containing evaluation metrics.
        """
        json.dump(json_data, open(out_json, 'w'), indent=2)

    @staticmethod
    def write_out_md(out_file: str, out_text: str):
        """Write evaluation summary to Markdown file.
        
        Args:
            out_file: Path to output Markdown file.
            out_text: Markdown formatted summary text.
        """
        open(out_file, 'w').write(out_text)
