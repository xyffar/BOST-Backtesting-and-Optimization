# config.py

import datetime as dt
from os.path import dirname, join

import streamlit as st
import yaml
from openpyxl.styles import Border, Font, PatternFill, Side

streamlit_obj = type[st.delta_generator.DeltaGenerator]

# FOLDER_STRATS = "strategies"

# # --- Global Default Parameters (only non-optimizable ones) ---
# INITIAL_CAPITAL = 10000.0  # Initial capital for the backtest
# COMMISSION_PERCENT = 0.002  # 0.2% commission per trade

# Default dates (last 5 years until today)
# Formatted as dd/mm/yyyy
DEFAULT_START_DATE = dt.datetime(dt.datetime.now().year - 5, 1, 1).strftime("%d/%m/%Y")
MIN_DATE = dt.datetime(dt.datetime.now().year - 50, 1, 1).strftime("%d/%m/%Y")
DEFAULT_END_DATE = dt.date.today().strftime("%d/%m/%Y")

# Available data granularity
# DATA_INTERVALS = ["1h", "1d", "1wk", "1mo"]

# Default ticker for S&P500 benchmark
# SP500_TICKER = (
#     "SPY"  # SPY is an ETF that tracks the S&P500, easier to download with yfinance
# )

# Mapping of pandas_ta moving average types
# MA_TYPES = {
#     "SMA": ta.sma,
#     "EMA": ta.ema,
#     "WMA": ta.wma,
#     "RMA": ta.rma,  # Also known as Wilders' Smoothing Average
# }

# Objective functions for optimization
# Metrics available for optimization (value: is_higher_better_bool)
# OPTIMIZATION_OBJECTIVES = {
#     'Return [%]': True,
#     'Sharpe Ratio': True,
#     'Sortino Ratio': True,
#     'Equity Final [$]': True,
#     'Max. Drawdown [%]': True, # Lower (less negative) is better
#     'Win Rate [%]': True,
#     'Profit Factor': True,
#     'SQN': True
# }

# OPTIMIZATION_OBJECTIVES = {
#     "Return [%]": True,
#     "Sharpe Ratio": True,
#     "Sortino Ratio": True,
#     "Max. Drawdown [%]": True,
#     "Max. Drawdown Duration": True,
#     "Win Rate [%]": True,
#     "SQN": True,
#     "# Trades": True,
# }

# DEFAULT_OPTIMIZATION_METRIC = "Return [%]"  # Default metric to optimize
# MAX_OPTIMIZATION_COMBINATIONS_WARNING = (
#     5000  # Warning threshold for combinations for Grid Search
# )
# MAX_OPTIMIZATION_COMBINATIONS_LIMIT = (
#     10000  # Hard limit for combinations for Grid Search to prevent crashes
# )

# # Optimization Methods
# OPTIMIZATION_METHODS = {
#     "Grid Search": "grid",
#     "SAMBO": "sambo",  # Bayesian Optimization
# }
# DEFAULT_OPTIMIZATION_METHOD = "Grid Search"

# # List of static input fields that remain visible in both modes
# STATIC_INPUT_FIELDS = [
#     "tickers",
#     "initial_capital",
#     "commission_percent",
#     "start_date",
#     "end_date",
#     "data_interval",
#     "selected_strategy",
# ]

# # --- Custom CSS for UI elements ---
# CUSTOM_CSS = """
# <style>
#     /* This selector is more specific for horizontal lines within the sidebar */
#     .stSidebar div[data-testid="stSidebarContent"] hr {
#         margin-top: 0px;  /* Top margin reduced to 0 */
#         margin-bottom: 0px; /* Bottom margin reduced to 0 */
#     }

#     /* Style for active mode button */
#     .mode-button.active {
#         background-color: #4F8BF9; /* Darker background color to indicate activation */
#         color: white; /* White text */
#         padding: 0.25rem 0.75rem; /* Similar padding to Streamlit buttons */
#         border-radius: 0.5rem; /* Rounded borders */
#         border: 1px solid #4F8BF9; /* Border with the same color as the background */
#         text-align: center;
#         font-weight: 600; /* Semi-bold */
#         cursor: default; /* Do not show pointer to indicate it's not clickable (already active) */
#         margin: 0; /* Remove default paragraph margins */
#         display: flex; /* Use flexbox to center text */
#         align-items: center;
#         justify-content: center;
#         height: 38px; /* Height to align with standard Streamlit buttons */
#         transition: background-color 0.3s, color 0.3s; /* Smooth transition */
#     }

#     /* Style for inactive mode buttons */
#     .stButton > button {
#         background-color: #26272e; /* Dark background for inactive buttons */
#         color: #FAFAFA; /* Light text */
#         border: 1px solid rgba(250, 250, 250, 0.2); /* Light border */
#         border-radius: 0.5rem;
#         height: 38px;
#         font-weight: 600;
#         transition: background-color 0.3s, color 0.3s, border-color 0.3s;
#     }
#     .stButton > button:hover {
#         border-color: #4F8BF9; /* Blue border on hover */
#         color: #4F8BF9; /* Blue text on hover */
#     }
# </style>
# """

# # --- Display Texts for UI elements ---
# DISPLAY_TEXTS = {
#     "app_title": "👑 Backtesting e Ottimizzazione di Strategie di Trading 👑",
#     "page_title": "BOST",
#     "page_icon": "👑",
#     "mode_backtest": "📊 Modalità Backtest",
#     "mode_optimization": "✨ Modalità Ottimizzazione",
#     "sidebar_choose_mode": "Scegli la modalità",
#     "button_backtest": "Backtest",
#     "button_optimization": "Ottimizzazione",
#     "ticker_input_label": "Ticker (separati da virgola)",
#     "initial_capital_label": "Capitale Iniziale",
#     "commission_percent_label": "Comm. (%)",
#     "data_period_granularity_header": "Periodo e Granularità Dati",
#     "start_date_label": "Data Inizio",
#     "end_date_label": "Data Fine",
#     "data_granularity_label": "Granularità Dati",
#     "strategy_selection_label": "Strategia",
#     "strategy_params_subheader": "Parametri della Strategia",
#     "run_backtest_button": "Avvia Backtest",
#     "optimization_objective_label": "Obiettivo",
#     "optimization_method_label": "Metodo",
#     "sambo_max_tries": "Simulazioni per SAMBO",
#     "optimization_params_subheader": "Parametri di Ottimizzazione",
#     "run_optimization_button": "Avvia Ottimizzazione",
#     "messages": {
#         "enter_ticker_error": "Inserisci almeno un ticker per avviare il backtest/ottimizzazione.",
#         "select_valid_strategy_error": "Seleziona una strategia valida.",
#         "define_optimization_ranges_error": "Definisci i range per i parametri di ottimizzazione.",
#         "downloading_benchmark": "Download in corso di {SP500_TICKER} (benchmark)...",
#         "download_success_benchmark": "Download eseguito per: {SP500_TICKER}",
#         "error_calculating_benchmark_stats": "Errore nel calcolo delle statistiche Buy & Hold per il benchmark ({SP500_TICKER}): {e}",
#         "benchmark_data_not_available": "Impossibile scaricare i dati per il benchmark S&P500. Il confronto non sarà disponibile.",
#         "downloading_ticker": "Download in corso di {ticker} (ticker {current_idx}/{total_tickers})...",
#         "download_success_ticker": "Download eseguito per: ",  # To be concatenated with tickers
#         "download_failed_ticker": "Download fallito per: ",  # To be concatenated with tickers
#         "execution_in_progress": "Backtesting in corso per ",
#         "execution_completed": "Esecuzione completata per: ",  # To be concatenated with tickers
#         "execution_failed": "Esecuzione fallita per: ",  # To be concatenated with tickers
#         "running_optimization": "Esecuzione ottimizzazione per {ticker} ({current_idx}/{total_tickers})... Questo potrebbe richiedere tempo!",
#         "no_results_to_show": "Nessun risultato da mostrare.",
#         "results_for_ticker": "Risultati per {ticker}",
#         "param_display_min": "Min",
#         "param_display_max": "Max",
#         "param_display_step": "Step",
#         "param_invalid_value": "Valore non valido per '{param_name}'. Utilizzo il default.",
#         "select_strategy_to_display_params": "Seleziona una strategia per visualizzare i suoi parametri.",
#         "optimization_options_header": "{display_name} (Opzioni per Ottimizzazione)",
#         "select_options_multiselect": "Seleziona Opzioni",
#         "param_type_not_supported_optimization": "Tipo di parametro non supportato per ottimizzazione: {param_name}",
#         "select_strategy_to_set_opt_params": "Seleziona una strategia per impostare i parametri di ottimizzazione.",
#         "table_combinations_header": "Tabella delle Combinazioni di Parametri e Valori Obiettivo",
#         "best_params_found": "Migliori Parametri Trovati",
#         "stats_best_combination": "Statistiche della Migliore Combinazione",
#         "optimization_heatmap": "Heatmap di Ottimizzazione",
#         "heatmap_not_available": "Heatmap non disponibile per questa ottimizzazione.",
#         "sambo_plots": "Grafici SAMBO",
#         "backtest_stats_benchmark_comparison": "Statistiche del Backtest e Confronto con Benchmark (SPY)",
#         "benchmark_comparison_not_available": "Impossibile generare confronto con benchmark.",
#         "interactive_backtest_chart": "Grafico Interattivo del Backtest",
#         "chart_display_error": "Grafico interattivo non visualizzato a causa di un errore nella generazione HTML o di un tipo di oggetto non gestito.",
#         "chart_not_generated": "Grafico interattivo non generato. Controlla la logica della strategia o i dati.",
#         "list_of_trades": "Lista dei Trade",
#         "no_trades_executed": "Nessun trade eseguito per questa strategia.",
#         "export_results_subheader": "Esporta Risultati",
#         "download_excel_button": "Scarica Report Excel",
#         "strategy_not_found": "Strategia non trovata: {strategy_name}",
#         "could_not_find_spec": "Impossibile trovare la specifica per il modulo {file_path}",
#         "error_loading_strategy": "Errore durante il caricamento della strategia '{strategy_name}' dal file '{file_path}': {e}",
#         "number_of_combinations_label": "Numero di combinazioni di parametri da provare",  # Added missing key here
#     },
#     "data_handler": {
#         "no_data_found": "Nessun dato trovato per {ticker} con il range e l'intervallo specificati.",
#         "required_columns_missing": "Colonne richieste (Open, High, Low, Close, Volume) mancanti per {ticker}.",
#         "data_download_success": "Dati scaricati con successo per {ticker}.",
#         "data_download_error": "Errore durante il download dei dati per {ticker}: {e}",
#         "no_benchmark_data_found": "Nessun dato benchmark trovato per {SP500_TICKER}.",
#         "benchmark_download_success": "Dati benchmark ({SP500_TICKER}) scaricati con successo.",
#         "benchmark_download_error": "Errore durante il download dei dati benchmark ({SP500_TICKER}): {e}",
#     },
#     "excel_exporter": {
#         "metric_column_name": "Metrica",
#         "no_comparison_data": "Dati di confronto non disponibili.",
#         "inf_value": "Inf",
#         "nan_value": "N/A",
#         "comparison_table_title": "Statistiche di Confronto Strategia vs. Benchmark",
#         "no_trades_executed_excel": "Nessun trade eseguito per questo ticker.",
#         "trades_table_title": "Dettagli dei Trade Eseguiti",
#         "no_results_sheet_name": "Nessun_Risultato",
#         "no_results_found": "Nessun risultato di backtest disponibile da esportare.",
#     },
#     "optimizer_runner": {
#         "no_data_for_optimization": "Impossibile eseguire l'ottimizzazione: dati non disponibili.",
#         "invalid_objective_function": "Funzione obiettivo '{key}' non valida per l'ottimizzazione.",
#         "check_config_format_simple": " Verifica che la sua definizione in config.py sia una chiave con un valore booleano (True/False).",
#         "invalid_objective_value_type": " Il valore per la funzione obiettivo '{key}' in config.py deve essere un booleano, non '{type}'.",
#         "optimization_method_not_supported": "Metodo di ottimizzazione '{method}' non supportato.",
#         "optimization_success": "Ottimizzazione completata con successo.",
#         "optimization_error": "Errore durante l'ottimizzazione: {e}. Controlla i parametri e i dati.",
#         "no_valid_combinations": "L'ottimizzazione non ha prodotto risultati validi, probabilmente a causa dei vincoli applicati o di range non adeguati.",
#         "no_valid_results": "L'ottimizzazione non ha prodotto risultati validi.",
#         "heatmap_col_missing": "La colonna '{col}' non è stata trovata nel heatmap dei risultati. "
#         "Potrebbe esserci un problema con la metrica o i risultati dell'ottimizzazione.",
#         "unexpected_param_range_type": "Tipo di range parametro inatteso per {param}: {type}.",
#         "heatmap_title": "Heatmap Ottimizzazione per {obj}",
#         "single_param_plot_title": "Risultati Ottimizzazione per {obj} vs. {param}",
#         "multi_param_plot_info": "Visualizzazione Heatmap non disponibile per più di 2 parametri ottimizzati.",
#         "heatmap_plot_error": "Errore durante la generazione del grafico heatmap: {e}",
#     },
# }

# Settings for Excel time logs
# Define a palette of distinct colors for table headers
HEADER_COLORS = [
    PatternFill(start_color="ADD8E6", end_color="ADD8E6", fill_type="solid"),  # Light Blue
    PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid"),  # Light Green
    PatternFill(start_color="FFDAB9", end_color="FFDAB9", fill_type="solid"),  # Peach
    PatternFill(start_color="FFC0CB", end_color="FFC0CB", fill_type="solid"),  # Pink
    PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid"),  # Light Gray
    PatternFill(start_color="FFFFE0", end_color="FFFFE0", fill_type="solid"),  # Light Yellow
    PatternFill(start_color="E0BBE4", end_color="E0BBE4", fill_type="solid"),  # Lavender
    PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid"),  # Light Pink
    PatternFill(start_color="FFDEAD", end_color="FFDEAD", fill_type="solid"),  # Navajo White
    PatternFill(start_color="AFEEEE", end_color="AFEEEE", fill_type="solid"),  # Pale Turquoise
]

# Row striping colors (lighter variations or just white/light gray)
ROW_STRIPE_COLORS = [
    PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid"),  # White
    PatternFill(start_color="F5F5F5", end_color="F5F5F5", fill_type="solid"),  # Lighter Gray
]

# Font for headers
HEADER_FONT = Font(bold=True, color="000000")  # Black text

# Border style (light grey thin border)
THIN_BORDER = Border(
    left=Side(style="thin", color="D3D3D3"),
    right=Side(style="thin", color="D3D3D3"),
    top=Side(style="thin", color="D3D3D3"),
    bottom=Side(style="thin", color="D3D3D3"),
)


def load_messages_from_yaml(file_path: str) -> dict:
    """Carica i messaggi testuali da un file YAML specificato."""
    try:
        with open(file_path, encoding="utf-8") as f:
            messages = yaml.safe_load(f)
            return messages if messages is not None else {}
    except FileNotFoundError:
        print(f"Errore: Il file dei messaggi YAML non trovato a '{file_path}'.")
        return {}
    except yaml.YAMLError as e:
        print(f"Errore durante il parsing del file YAML '{file_path}': {e}")
        return {}
    except Exception as e:
        print(f"Si è verificato un errore inatteso durante il caricamento di '{file_path}': {e}")
        return {}


# Carica i messaggi dal file YAML
MESSAGES = load_messages_from_yaml(join(dirname(__file__), "messages.yaml"))
