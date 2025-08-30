from datetime import datetime

import pandas as pd
from st_aggrid import AgGrid, ColumnsAutoSizeMode, GridOptionsBuilder, JsCode


class DoubleClickTable:
    """An interactive table component for displaying pandas DataFrames with double-click event handling.

    Attributes
    ----------
    _original_df : pd.DataFrame
        The original DataFrame provided to the table.
    _df : pd.DataFrame
        The processed DataFrame used for display.
    _key : str
        Unique identifier for the table instance.
    _height : int
        Maximum height of the table in pixels.
    _allow_unsafe_jscode : bool
        Whether to allow execution of unsafe JavaScript code.
    _grid_response : Any
        Stores the response from AgGrid.
    _recently_double_clicked : bool
        Indicates if a double-click event occurred recently.
    _double_click_row_index : int or None
        Index of the last double-clicked row.
    _double_clicked_table : str or None
        Table name of the last double-clicked row.
    _mixed_columns : bool
        Indicates if the table has columns with mixed data types.
    _js_double_click_handler : JsCode or None
        JavaScript handler for double-click events.

    Methods
    -------
    display()
        Renders the interactive table in the UI.
    get_double_clicked_data()
        Processes information about the most recent double-click event.
    data
        Returns the current data displayed in the table.
    is_double_clicked
        Checks if a row was double-clicked recently.
    double_clicked_table
        Returns the table name of the last double-clicked row.
    idx_last_double_clicked
        Returns the index of the last double-clicked row.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        key: str = "default_table",
        max_height: int = 300,
        allow_unsafe_jscode: bool = True,
        double_clickable: bool = True,
        mixed_columns: bool = False,
    ) -> None:
        """Initialize a DoubleClickTable with the provided DataFrame and configuration options.

        Args:
            df (pd.DataFrame): The DataFrame displayed in the interactive table.
            key (str, optional): Unique identifier for the table instance. Defaults to "default_table".
            max_height (int, optional): Maximum height of the table in pixels. Defaults to 300.
            allow_unsafe_jscode (bool, optional): Whether to allow execution of unsafe JavaScript code.
                Defaults to True.
            double_clickable (bool, optional): Whether the table should handle double-click events.
                Defaults to True.
            mixed_columns (bool, optional): Whether the table has columns with mixed data types.
                Defaults to False.

        Raises:
            ValueError: If the input 'df' is not a pandas DataFrame.

        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input 'df' must be a pandas DataFrame.")

        # Handle multi-index columns and store the original DataFrame
        self._original_df = df.copy()  # Store the original DataFrame
        self._df = self._handle_multi_index_columns(df.copy()).round(2)  # Handle multi-index columns

        self._key = key  # Unique identifier for the table instance
        self._height = min(max_height, int(31.3 * len(self._df)) + 1)  # Ensure height is at least 31.25 pixels per row
        self._allow_unsafe_jscode = allow_unsafe_jscode
        self._grid_response = None  # Store the response from AgGrid
        self._recently_double_clicked = False
        self._double_click_row_index = None
        self._double_clicked_table = None
        self._mixed_columns = mixed_columns

        # Define the JavaScript handler for double-click events
        self._js_double_click_handler = (
            JsCode(f"""
        function(params) {{
            // Get the index and table name of the clicked row
            let clickedRowIndex = params.rowIndex;
            let clickedTable = '{self._key}';

            // Set the clicked row index, table name, and timestamp in the node data
            params.node.setDataValue('Idx', clickedRowIndex);
            params.node.setDataValue('Table', clickedTable);
            params.node.setDataValue('DoubleClickTimestamp', Date.now());
            }}
        """)
            if double_clickable
            else None
        )

    def _handle_multi_index_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle multi-index columns by converting them to single-level columns with underscores.

        Args:
            df (pd.DataFrame): The DataFrame with multi-index columns.

        Returns:
            pd.DataFrame: The DataFrame with single-level columns.

        """
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(col) for col in df.columns]
        return df

    def _build_column_defs(self) -> dict[str, dict[str, list | str]]:
        column_defs = {}
        if isinstance(self._original_df.columns, pd.MultiIndex):
            top_level_names = self._original_df.columns.get_level_values(0).unique()
            for top_level in top_level_names:
                group = {f"{top_level}": {"headerName": top_level, "children": []}}
                for col_name in self._original_df.columns:
                    if col_name[0] == top_level:
                        field_name = "_".join(col_name)
                        child_col = {"headerName": col_name[1], "field": field_name}
                        group[f"{top_level}"]["children"].append(child_col)
                column_defs = {**column_defs, **group}
        else:
            for col in self._df.columns:
                column_defs = {**column_defs, **{f"{col}": {"headerName": col, "field": col}}}

        # Aggiunge le colonne nascoste per la gestione del double-click
        column_defs = {
            **column_defs,
            **{"Idx": {"headerName": "Idx", "field": "Idx", "hide": True}},
            **{"Table": {"headerName": "Table", "field": "Table", "hide": True}},
            **{
                "DoubleClickTimestamp": {
                    "headerName": "DoubleClickTimestamp",
                    "field": "DoubleClickTimestamp",
                    "hide": True,
                }
            },
        }

        return column_defs

    def _build_grid_options(self) -> dict:
        """Costruisce le opzioni della griglia."""
        gb = GridOptionsBuilder.from_dataframe(self._df)

        if self._mixed_columns:
            for col in self._df.columns:
                gb.configure_column(col, cellDataType=False)

        # Genera le definizioni delle colonne con o senza gruppi
        column_defs = self._build_column_defs()
        gb.configure_grid_options(columnDefs=column_defs)

        # Configura l'evento onCellDoubleClicked
        gb.configure_grid_options(onCellDoubleClicked=self._js_double_click_handler)

        gb.configure_grid_options(
            autoSizeStrategy={"type": "fitCellContents", "skipHeader?": True},
            # onFirstDataRendered=self._js_code_autofit_columns,  # Inietta il codice JS nell'evento onGridReady
        )

        # Configurazioni comuni
        gb.configure_selection("single", use_checkbox=False)
        gb.configure_pagination(enabled=False)
        gb.configure_default_column(editable=False, filter=True, sortable=True)

        return gb.build()

    def display(self) -> None:
        """Display the interactive data grid using AgGrid with the specified configuration.

        Renders the DataFrame (`self._df`) in an interactive grid within the UI, applying the provided grid options,
        sizing, and update mode. After rendering, it processes any double-clicked data by invoking `get_double_clicked_data()`.

        Returns:
            None.

        """
        self._grid_response = AgGrid(
            self._df,
            gridOptions=self._build_grid_options(),
            # fit_columns_on_grid_load=True,
            allow_unsafe_jscode=self._allow_unsafe_jscode,
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
            update_on=["selectionChanged"],
            height=self._height,
            width="100%",
            reload_data=False,
            key=self._key,
        )

        self.get_double_clicked_data()

    @property
    def data(self) -> pd.DataFrame:
        """Returns the current data displayed in the interactive table as a pandas DataFrame.

        If a grid response exists, returns the 'data' from the grid response; otherwise, returns the original DataFrame.

        Returns:
            pd.DataFrame: The data currently shown in the interactive table.

        """
        return self._grid_response["data"] if self._grid_response else self._df

    def get_double_clicked_data(self) -> None:
        """Extract and process information about the most recent double-click event in the table data.

        This method checks if the DataFrame `self.data` contains the columns "Idx", "Table", and "DoubleClickTimestamp".
        If so, it identifies the row with the most recent double-click event based on the maximum value in the
        "DoubleClickTimestamp" column. It then updates the following instance attributes:
            - self._double_click_row_index: Index of the row with the most recent double-click.
            - self._double_clicked_table: Value from the "Table" column at the double-clicked row.
            - self._recently_double_clicked: Boolean indicating if the double-click occurred within the last second.

        Assumes that "DoubleClickTimestamp" values are in milliseconds since the epoch.
        """
        if {"Idx", "Table", "DoubleClickTimestamp"}.issubset(self.data.columns):
            last_double_click = self.data["DoubleClickTimestamp"].max()
            self._double_click_row_index = self.data["DoubleClickTimestamp"].idxmax()
            self._double_clicked_table: str | None = self.data.at[self._double_click_row_index, "Table"]
            now = int(datetime.now().timestamp() * 1000)
            self._recently_double_clicked: bool = (now - last_double_click) < 2000

    @property
    def is_double_clicked(self) -> bool:
        """Check if a row was double-clicked recently."""
        return self._recently_double_clicked and self._double_clicked_table == self._key

    @property
    def double_clicked_table(self) -> str | None:
        """Returns the table name of the last double-clicked row."""
        return self._double_clicked_table if self.is_double_clicked else ""

    @property
    def idx_last_double_clicked(self) -> int | str | None:
        """Returns the index of the last double-clicked row."""
        return self._double_click_row_index if self.is_double_clicked else -1
