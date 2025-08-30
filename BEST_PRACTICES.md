# 📘 Project Best Practices

## 1. Project Purpose
A Streamlit-based platform for backtesting and optimizing algorithmic trading strategies. It integrates the backtesting.py engine, provides a modular strategy framework, supports parameter optimization (Grid Search and SAMBO), and includes Monte Carlo analysis, benchmark comparison, and Excel reporting.

## 2. Project Structure
- Root
  - bost.py: Streamlit entry point using multipage navigation (pages/01_backtest_mode.py, pages/02_optimization_mode.py)
  - requirements.txt, pyproject.toml: runtime and tooling config
  - tests/: pytest-based unit tests
  - strategies/: strategy classes extending a shared base (CommonStrategy)
  - src/
    - calcs/: backtest, optimizer, Monte Carlo, metrics
    - config/: session state, defaults, YAML-driven messages/config
    - data_handlers/: data fetch, Excel export, logging
    - ui_components/: UI building blocks and results rendering
    - utils/: helpers (session state management, strategy discovery, optimization utility)
  - assets/, outputs/, pages/: static assets, generated files, and Streamlit pages

Key files/directories
- strategies/common_strategy.py: Base class for all strategies (Stop Loss/Take Profit, helpers for buy/close)
- strategies/strategy_*.py: Concrete strategies with PARAMS_INFO metadata, DISPLAY_NAME, optional optimization_constraint
- src/config/config.py: Central Streamlit session state schema (ss), default dates, YAML loader for MESSAGES
- src/config/messages.yaml: UI labels, optimization settings, stats mapping, CSS, Monte Carlo settings
- src/calcs/backtest_runner.py: Orchestrates per-ticker backtests, plot creation, benchmark comparisons
- src/calcs/optimizer_runner.py: Grid/SAMBO optimization, result aggregation, ranking, optional MC on top combos
- src/calcs/monte_carlo.py: Vectorized Monte Carlo sampling and stats aggregation
- src/ui_components/display_results.py: Unified results renderer (backtest/optimization)
- src/ui_components/backtest_ui.py, optimization_ui.py, ui.py: UI controls for parameters, optimization, and sidebar input

Entry points and config
- Run app: streamlit run bost.py (or bost.bat)
- Messages/labels/metrics are configured via src/config/messages.yaml and loaded at runtime

## 3. Test Strategy
- Framework: pytest
- Location: tests/
- Conventions:
  - Test strategy logic in isolation by mocking backtesting.py interactions where needed
  - Use pytest.mark.parametrize for scenario coverage (happy paths and edges)
  - Validate public API shape for strategies: DISPLAY_NAME, PARAMS_INFO, optimization_constraint
  - Assert behavioral decisions (buy/close/no action) using MagicMock on helper methods (e.g., _buy_long, _close_position)
- Mocking guidelines:
  - Patch strategy.I to a pure function for deterministic indicator series
  - Replace external calls (network/data/UI) with mocks
- Scope:
  - Unit tests for strategy next/init
  - Unit tests for PARAMS_INFO structure and constraints
  - Optional integration tests for runners can use small synthetic data

## 4. Code Style
- Tooling: Ruff (lint + format)
  - target-version: py310; line-length: 120; double quotes; pep8-naming; annotations; pyupgrade; bugbear
  - Ignored: E501 (handled by line-length), D100/D104 (module/package docstrings), ANN101/ANN102 (self/cls hints)
- Typing:
  - Use explicit type hints (PEP 484). Prefer pandas.Series, numpy.ndarray, and precise dict types
  - Use ClassVar for class-level constants (e.g., DISPLAY_NAME, PARAMS_INFO)
- Async: not used; avoid unnecessary async
- Naming:
  - Files: strategy_*.py for concrete strategies; snake_case for modules; PascalCase for classes
  - Parameters/variables: snake_case
- Docstrings/Comments:
  - Short, action-oriented docstrings for public functions/classes; include Args/Returns when non-trivial
  - Keep inline comments concise and value-adding
- Imports:
  - Absolute imports within repo (e.g., from src.config.config import MESSAGES)
  - Group stdlib, third-party, local; rely on Ruff for ordering
- Error handling:
  - Validate inputs early; prefer informative messages via MESSAGES where relevant to UI
  - Fail fast for impossible states; use st.error/warning in UI layers only

## 5. Common Patterns
- Strategy architecture
  - Inherit from CommonStrategy (extends backtesting.Strategy)
  - Implement init and next; use self.I to register indicators
  - Use helper methods _buy_long() and _close_position() to centralize SL/TP and closing behavior
  - Expose:
    - DISPLAY_NAME: str
    - PARAMS_INFO: list[dict]; include all tunables and sl_pct/tp_pct
    - optimization_constraint: Callable[[pd.Series], bool] | None for optimizer filtering
- PARAMS_INFO conventions
  - Keys: name, type, default, lowest, highest, min, max, step; optional options for categorical
  - Percent params (sl_pct, tp_pct) are displayed as percentages in UI but stored as decimals
- Configuration and state
  - Text, metrics, and options via MESSAGES (YAML); never hardcode UI strings in code
  - Use ss (Streamlit session_state) keys defined in session_state_names; update/reset via utils helpers
- Optimization
  - Grid: translate UI ranges into lists (int ranges, float linspace, categorical lists)
  - SAMBO: optional Bayesian optimization (sambo library)
  - Enforce strategy-specific optimization_constraint when present
- Monte Carlo
  - Sampling methods configured in YAML: resampling_con_reimmissione (bootstrap) and permutazione (permutation)
  - Vectorized sampling and metrics; store plots and tables into session state for UI rendering
- UI
  - Streamlit fragments and DoubleClickTable for interactive tables
  - Results separated by mode (backtest vs optimization); tabs per ticker
  - Excel export via data_handlers/excel_exporter with stylized sheets

## 6. Do's and Don'ts
- Do
  - Add new strategies under strategies/ as strategy_<name>.py; subclass CommonStrategy
  - Provide PARAMS_INFO, DISPLAY_NAME, and an optimization_constraint if needed
  - Keep indicators vectorized; prefer pandas/numpy operations
  - Respect session_state contract; reset via utils.reset_* helpers before runs
  - Reuse MESSAGES for all user-facing text; extend YAML when adding features
  - Write unit tests for new strategy logic and metadata
  - Keep functions pure where possible (return values over global mutation), except UI rendering functions
- Don't
  - Don’t manipulate st.session_state ad hoc; use centralized helpers and defined keys
  - Don’t hardcode labels, metrics, or constants already defined in YAML
  - Don’t introduce side effects in compute layers (calcs/*) that belong to UI layers
  - Don’t bypass _buy_long/_close_position; keep SL/TP logic consistent
  - Don’t assume external data availability; guard for empty/None dataframes and handle gracefully

## 7. Tools & Dependencies
- Key libraries
  - backtesting.py: backtest engine
  - streamlit (+ streamlit-bokeh): UI and interactive plots
  - pandas, numpy, pandas-ta: data manipulation and indicators
  - yfinance: data retrieval
  - matplotlib: plotting
  - xlsxwriter/openpyxl: Excel export and styling
  - sambo: advanced optimization
- Setup
  - Python 3.12.x recommended (see requirements.txt header)
  - pip install -r requirements.txt
  - Run: streamlit run bost.py (or execute bost.bat on Windows)

## 8. Other Notes
- Metrics mapping and naming are driven by MESSAGES['all_stats_properties']; if you add a metric, define its nickname and vector_function where applicable
- MultiIndex columns in results tables follow a strict scheme: ("Strategy", ...), ("BT Stats", ...), ("Benchmark Stats", ...), ("Bench. Comp. [%]", ...), ("Monte Carlo", ...)
- UI percent inputs for commission/sl_pct/tp_pct are shown in percent but converted to decimals for computation
- Strategy discovery relies on filenames starting with strategy_ and a valid DISPLAY_NAME attribute
- Keep operations memory-aware when handling many simulations/combinations; prefer vectorization and avoid large Python loops
