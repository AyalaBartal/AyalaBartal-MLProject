import os
from pathlib import Path


class ExecutableLocator:

    def get_dot_executable(self):
        dot_env = os.getenv("GRAPHVIZ_DOT")
        return Path(dot_env)
