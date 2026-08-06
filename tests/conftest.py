from pathlib import Path

from pytest import Config


def pytest_configure(config: Config) -> None:
    if "benchmark.py" not in {Path(arg.partition("::")[0]).name for arg in config.args}:
        config.option.benchmark_disable = True
