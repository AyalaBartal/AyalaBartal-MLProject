import json


class FileWriter:

    @staticmethod
    def write_out_json(out_json, json_data):
        json.dump(json_data, open(out_json, 'w'), indent=2)

    @staticmethod
    def write_out_md(out_file, out_text):
        open(out_file, 'w').write(out_text)
