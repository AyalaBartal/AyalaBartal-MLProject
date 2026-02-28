import os
from pathlib import Path


class ExecutableLocator:
    @staticmethod
    def get_dot_executable():
        dot_env = os.getenv("GRAPHVIZ_DOT")
        if dot_env:
            dot_path = Path(dot_env)
            if not dot_path.exists():
                raise FileNotFoundError('GRAPHVIZ_DOT is set but file does not exist: {}'.format(dot_path))
            if not os.path.isfile(dot_path):
                raise FileNotFoundError('GRAPHVIZ_DOT is set but its not a file: {}'.format(dot_path))
            return str(dot_path)
