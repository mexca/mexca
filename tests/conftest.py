"""Configure custom pytest markers for skipping tests on specific
platforms and environments (e.g., GitHub action runners).
"""

import platform

import pytest


def pytest_addoption(parser):
    """Add command line option for `env` to parser."""
    parser.addoption(
        "-E",
        action="store",
        metavar="NAME",
        help="Only run tests matching the environment NAME",
    )


def pytest_configure(config):
    """Register additional markers for `env` and `os`."""
    config.addinivalue_line(
        "markers", "skip_env(name): mark test to skip on named environment"
    )
    config.addinivalue_line(
        "markers", "run_env(name): mark test to run only on named environment"
    )
    config.addinivalue_line(
        "markers", "skip_os(name): mark test to run only on named os"
    )


def pytest_runtest_setup(item):
    """Define actions when markers are set."""
    skip_envnames = [
        mark.args[0] for mark in item.iter_markers(name="skip_env")
    ]
    run_envnames = [mark.args[0] for mark in item.iter_markers(name="run_env")]
    osnames = [mark.args[0] for mark in item.iter_markers(name="skip_os")]

    if skip_envnames and osnames and run_envnames:
        if (
            item.config.getoption("-E") in skip_envnames
            and item.config.getoption("-E") not in run_envnames
            and platform.system() in osnames[0]
        ):
            pytest.skip(
                f"Test skipped because env in {skip_envnames} and os in {osnames[0]}"
            )

    elif (
        item.config.getoption("-E")
        and item.config.getoption("-E") not in run_envnames
    ):
        pytest.skip(f"Test skipped because env NOT in {run_envnames}")

    elif skip_envnames and item.config.getoption("-E") in skip_envnames:
        pytest.skip(f"Test skipped because env in {skip_envnames}")

    elif osnames and platform.system() in osnames[0]:
        pytest.skip(f"Test skipped because os in {osnames[0]}")
