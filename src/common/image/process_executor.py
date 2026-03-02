import subprocess
from typing import Sequence, Optional


# Thin wrapper around subprocess.run to simplify unit testing.
class ProcessExecutor:

    # Execute a process. Parameters mirror subprocess.run for flexibility.
    def run(self,
        args: Sequence[str],
        check: bool = True,
        capture_output: bool = False,
        text: bool = False,
        timeout: Optional[float] = None,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            args,
            check=check,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
        )