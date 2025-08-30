import datetime as dt
import re
import sys
from types import ModuleType
from typing import Any

import pytest

from src.utils.utils import format_date


def test_format_date_single_digit_day_month_padding() -> None:
    """Test that single-digit day and month are padded with a zero."""
    date_obj = dt.date(2023, 3, 7)
    assert format_date(date_obj) == "07/03/2023"


def test_format_date_leap_day() -> None:
    """Test that a leap day is formatted correctly."""
    date_obj = dt.date(2020, 2, 29)
    assert format_date(date_obj) == "29/02/2020"


def test_format_date_returns_str_and_pattern() -> None:
    """Test that the function returns a string in the correct format."""
    date_obj = dt.date(1999, 12, 1)
    result = format_date(date_obj)
    assert isinstance(result, str)
    assert re.fullmatch(r"\d{2}/\d{2}/\d{4}", result)
    assert result == "01/12/1999"


def test_format_date_boundary_years() -> None:
    """Test that boundary years are formatted correctly."""
    assert format_date(dt.date.min) == "01/01/0001"
    assert format_date(dt.date.max) == "31/12/9999"


def test_format_date_with_datetime_subclass() -> None:
    """Test that a datetime object is formatted correctly."""
    datetime_obj = dt.datetime(2024, 8, 15, 13, 45, 59)
    assert format_date(datetime_obj) == "15/08/2024"


@pytest.mark.parametrize("invalid_input", [None, 123])
def test_format_date_invalid_type_raises(invalid_input: Any) -> None:
    """Test that an invalid input type raises an AttributeError."""
    with pytest.raises(AttributeError):
        format_date(invalid_input)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def utils_env() -> ModuleType:
    """Set up a mock environment for testing utils."""
    # Create dummy external modules to allow importing utils without full dependencies
    backtesting_mod = ModuleType("backtesting")
    backtesting_backtesting_mod = ModuleType("backtesting.backtesting")

    class DummyStrategy:
        pass

    backtesting_backtesting_mod.Strategy = DummyStrategy
    backtesting_mod.backtesting = backtesting_backtesting_mod
    sys.modules.setdefault("backtesting", backtesting_mod)
    sys.modules.setdefault("backtesting.backtesting", backtesting_backtesting_mod)

    streamlit_mod = ModuleType("streamlit")

    def dummy_warning(msg: str) -> None:  # noqa: ANN001
        return None

    streamlit_mod.warning = dummy_warning
    sys.modules.setdefault("streamlit", streamlit_mod)

    # Strategies package and CommonStrategy
    strategies_pkg = ModuleType("strategies")
    sys.modules.setdefault("strategies", strategies_pkg)
    common_strategy_mod = ModuleType("strategies.common_strategy")

    class CommonStrategy:
        DISPLAY_NAME = "Common Strategy"
        sl_pct = 0.05
        tp_pct = 0.10

    common_strategy_mod.CommonStrategy = CommonStrategy
    sys.modules.setdefault("strategies.common_strategy", common_strategy_mod)

    # src.config.config with required names
    src_pkg = ModuleType("src")
    sys.modules.setdefault("src", src_pkg)
    config_pkg = ModuleType("src.config")
    sys.modules.setdefault("src.config", config_pkg)
    config_mod = ModuleType("src.config.config")
    config_mod.MESSAGES = {
        "general_settings": {
            "folder_strategies": "strategies",
            "base_strategy_filename": "common_strategy.py",
        }
    }
    config_mod.ss = {}
    config_mod.session_state_names = {}
    config_mod.bt_stats = object()
    sys.modules.setdefault("src.config.config", config_mod)

    import importlib as py_importlib

    utils_module = py_importlib.import_module("src.utils.utils")
    return utils_module


def test_calculate_optimization_combinations_mixed_types(utils_env: ModuleType, mocker: Any) -> None:
    """Test that optimization combinations are calculated correctly for mixed types."""
    utils = utils_env
    # Ensure proper messages
    mocker.patch(
        "src.config.config.MESSAGES",
        {"general_settings": {"folder_strategies": "strategies", "base_strategy_filename": "common_strategy.py"}},
    )

    mocker.patch(
        "src.utils.utils.os.listdir",
        return_value=["strategy_beta.py", "helper.py", "common_strategy.py", "strategy_alpha.py"],
    )

    from types import ModuleType

    def import_side_effect(name: str) -> ModuleType:
        mod = ModuleType(name)
        if name == "strategies.strategy_alpha":

            class Alpha(utils.CommonStrategy):  # type: ignore[attr-defined]
                DISPLAY_NAME = "Alpha Z"

            mod.Alpha = Alpha
        elif name == "strategies.strategy_beta":

            class Beta(utils.Strategy):  # type: ignore[attr-defined]
                DISPLAY_NAME = "Beta A"

            mod.Beta = Beta
        return mod

    mocker.patch("src.utils.utils.importlib.import_module", side_effect=import_side_effect)

    result = utils.load_strategies()
    assert list(result.keys()) == ["Alpha Z", "Beta A"]
    assert issubclass(result["Alpha Z"], utils.CommonStrategy)  # type: ignore[attr-defined]
    assert issubclass(result["Beta A"], utils.Strategy)  # type: ignore[attr-defined]


def test_record_all_optimizations_records_multiple_runs(utils_env: ModuleType, mocker: Any) -> None:
    """Test that all optimization runs are recorded."""
    utils = utils_env
    mocker.patch(
        "src.config.config.MESSAGES",
        {"general_settings": {"folder_strategies": "strategies", "base_strategy_filename": "common_strategy.py"}},
    )
    mocker.patch(
        "src.utils.utils.os.listdir",
        return_value=["strategy_one.py", "strategy_two.py"],
    )

    def import_side_effect(name: str) -> ModuleType:
        mod = ModuleType(name)
        if name.endswith("strategy_one"):

            class One(utils.CommonStrategy):  # type: ignore[attr-defined]
                DISPLAY_NAME = "One"

            mod.One = One
        elif name.endswith("strategy_two"):

            class Two(utils.Strategy):  # type: ignore[attr-defined]
                DISPLAY_NAME = "Two"

            mod.Two = Two
        return mod

    mocker.patch("src.utils.utils.importlib.import_module", side_effect=import_side_effect)

    result = utils.load_strategies()
    assert len(result) == 2
    assert list(result.keys()) == ["One", "Two"]


def test_add_benchmark_comparison_adds_columns_and_diffs(utils_env: ModuleType, mocker: Any, capsys: Any) -> None:
    """Test that benchmark comparison columns are added correctly."""
    utils = utils_env
    mocker.patch(
        "src.config.config.MESSAGES",
        {"general_settings": {"folder_strategies": "strategies", "base_strategy_filename": "common_strategy.py"}},
    )
    mocker.patch(
        "src.utils.utils.os.listdir",
        return_value=["strategy_no_display.py", "strategy_valid.py"],
    )

    def import_side_effect(name: str) -> ModuleType:
        mod = ModuleType(name)
        if name.endswith("strategy_no_display"):

            class NoDisplay(utils.CommonStrategy):  # type: ignore[attr-defined]
                pass  # No DISPLAY_NAME attribute

            mod.NoDisplay = NoDisplay
        elif name.endswith("strategy_valid"):

            class Valid(utils.Strategy):  # type: ignore[attr-defined]
                DISPLAY_NAME = "Valid"

            mod.Valid = Valid
        return mod

    mocker.patch("src.utils.utils.importlib.import_module", side_effect=import_side_effect)

    result = utils.load_strategies()
    captured = capsys.readouterr()
    assert "does not have a valid 'DISPLAY_NAME' attribute" in captured.out
    assert "strategies.strategy_no_display" in captured.out
    assert list(result.keys()) == ["Valid"]


def test_record_all_optimizations_handles_exception_and_restores_run(
    utils_env: ModuleType, mocker: Any, capsys: Any
) -> None:
    """Test that exceptions are handled and the run method is restored."""
    utils = utils_env
    mocker.patch(
        "src.config.config.MESSAGES",
        {"general_settings": {"folder_strategies": "strategies", "base_strategy_filename": "common_strategy.py"}},
    )
    mocker.patch(
        "src.utils.utils.os.listdir",
        return_value=["strategy_broken.py", "strategy_ok.py"],
    )

    def import_side_effect(name: str) -> ModuleType:
        if name.endswith("strategy_broken"):
            raise Exception("boom")
        mod = ModuleType(name)

        class OK(utils.CommonStrategy):  # type: ignore[attr-defined]
            DISPLAY_NAME = "OK"

        mod.OK = OK
        return mod

    mocker.patch("src.utils.utils.importlib.import_module", side_effect=import_side_effect)

    result = utils.load_strategies()
    captured = capsys.readouterr()
    assert "Errore durante il caricamento" in captured.out
    assert "strategy_broken.py" in captured.out
    assert list(result.keys()) == ["OK"]


def test_calculate_optimization_combinations_zero_step_or_single_point(
    utils_env: ModuleType, mocker: Any
) -> None:
    """Test that zero step or single point combinations are calculated correctly."""
    utils = utils_env
    mocker.patch(
        "src.config.config.MESSAGES",
        {"general_settings": {"folder_strategies": "strategies", "base_strategy_filename": "common_strategy.py"}},
    )
    mocker.patch(
        "src.utils.utils.os.listdir",
        return_value=["common_strategy.py", "strategy_valid.py"],
    )

    def import_side_effect(name: str) -> ModuleType:
        mod = ModuleType(name)

        class Valid(utils.CommonStrategy):  # type: ignore[attr-defined]
            DISPLAY_NAME = "Valid"

        mod.Valid = Valid
        return mod

    import_mock = mocker.patch("src.utils.utils.importlib.import_module", side_effect=import_side_effect)

    result = utils.load_strategies()
    # Ensure base strategy file is ignored and not imported
    assert all(call.args[0] != "strategies.common_strategy" for call in import_mock.call_args_list)
    assert list(result.keys()) == ["Valid"]


def test_add_benchmark_comparison_zero_benchmark_yields_zero_diff(utils_env: ModuleType, mocker: Any) -> None:
    """Test that a zero benchmark yields a zero difference."""
    utils = utils_env
    mocker.patch(
        "src.config.config.MESSAGES",
        {"general_settings": {"folder_strategies": "strategies", "base_strategy_filename": "common_strategy.py"}},
    )
mocker.patch("src.utils.utils.os.listdir", return_value=["helper.txt", "readme.md"])
import_mock = mocker.patch("src.utils.utils.importlib.import_module")
result = utils.load_strategies()
assert result == {}
import_mock.assert_not_called()
