"""Configure custom pytest markers for skipping tests on specific
platforms and environments (e.g., GitHub action runners).
"""

import platform
import pytest
from huggingface_hub import HfApi


def pytest_addoption(parser):
    """Add command line option for `env` to parser.
    """
    parser.addoption(
        "-E",
        action="store",
        metavar="NAME",
        help="Only run tests matching the environment NAME",
    )

    parser.addoption(
        "-T",
        action="store",
        metavar="HF_TOKEN",
        type=str,
        help="Only run tests matching the environment NAME",
    )


def pytest_configure(config):
    """Register additional markers for `env` and `os`.
    """
    config.addinivalue_line(
        "markers", "skip_env(name): mark test to run only on named environment"
    )
    config.addinivalue_line(
        "markers", "skip_os(name): mark test to run only on named os"
    )


def pytest_sessionstart(session):
    auth_token = session.config.getoption("-T")
    api = HfApi()
    api.set_access_token(auth_token)


def pytest_runtest_setup(item):
    """Define actions when markers are set.
    """
    envnames = [mark.args[0] for mark in item.iter_markers(name="skip_env")]
    osnames = [mark.args[0] for mark in item.iter_markers(name="skip_os")]
    if envnames and osnames:
        if item.config.getoption("-E") in envnames and platform.system() in osnames[0]:
            pytest.skip(f"Test skipped because env in {envnames} and os in {osnames[0]}")

    elif envnames:
        if item.config.getoption("-E") in envnames:
            pytest.skip(f"Test skipped because env in {envnames}")

    elif osnames:
        if platform.system() in osnames[0]:
            pytest.skip(f"Test skipped because os in {osnames[0]}")
