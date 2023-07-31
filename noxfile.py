import os
from pathlib import Path

import nox

# Default sessions to run if no session handles are passed
nox.options.sessions = ["lock"]


DIR = Path(__file__).parent.resolve()


@nox.session(reuse_venv=True)
def lock(session: nox.Session) -> None:
    """
    Build a lock file with pip-tools

    Examples:

        $ nox --session lock
    """
    session.install("--upgrade", "pip", "setuptools", "wheel")
    session.install("--upgrade", "pip-tools")
    requirements_file = DIR / "requirements.txt"

    session.run(
        "pip-compile",
        "--resolver=backtracking",
        "--generate-hashes",
        "--output-file",
        f"{DIR / 'requirements.lock'}",
        requirements_file,
    )
