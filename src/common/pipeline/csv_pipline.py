import os

import pandas as pd
from typing import Callable, List, Tuple


class CsvPipline:

    # Type hint for converters and FilePair
    Converter = Callable[[pd.DataFrame], pd.DataFrame]
    StepPair = Tuple[str, Converter]

    # Run a pipeline of converters on DataFrame. Read from csv and write to csv at each step from start to end.
    def __init__(self, csv_temp_dir):
        self.csv_temp_dir = csv_temp_dir

    # Run a pipeline of converters on DataFrame. Read from csv and write to csv at each step all in dir csv_dir.
    def run(
            self,
            input_csv: str,
            output_csv: str,
            steps: List[StepPair]
    ):
        # Step 1: Load CSV
        data = pd.read_csv(input_csv)

        # Step 2: converter chain
        for step_name, converter in steps:
            print('Start step {}'.format(step_name))
            data = converter(data)
            self.validate_step_data(step_name, data)
            csv_name = '{}.csv'.format(step_name)
            step_csv_path = os.path.join(self.csv_temp_dir, csv_name)
            data.to_csv(step_csv_path, index=False)
            print('End step {}'.format(step_name))

        # Step 3: save final output
        data.to_csv(output_csv, index=False)

    @staticmethod
    def validate_step_data(step_name, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                f"step {step_name} must return a DataFrame, "
                f"got {type(data)}"
            )
