import sqlite3
import yfinance as yf


def download_example(
    db_path: str, ticker: str, period: str, interval: str, flush: bool = True
) -> None:
    """
    Downloads historical stock data for a given ticker and stores it in a SQLite database.

    This function creates or connects to a SQLite database at the specified path,
    downloads stock data using Yahoo Finance API, and stores it in a table named 'data'.
    If flush is True, all existing tables in the database will be dropped before creating
    the new data table.

    Parameters:
    -----------
    db_path : str
        Path to the SQLite database file
    ticker : str
        Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
    period : str
        The period for which to download data (e.g., '1y' for 1 year)
    interval : str
        The interval between data points (e.g., '1d' for daily data)
    flush : bool, default=True
        Whether to clear existing tables in the database before inserting new data

    Returns:
    --------
        None: The function modifies the database file at db_path but does not return any value

    Notes:
    ------
    The created table has the following schema:
    - id: INTEGER PRIMARY KEY
    - timestamp: REAL (Unix timestamp)
    - open: REAL (Opening price)
    - high: REAL (Highest price)
    - low: REAL (Lowest price)
    - close: REAL (Closing price)
    - volume: REAL (Trading volume)
    """
    con = sqlite3.connect(db_path)
    cursor = con.cursor()
    if flush:
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            con.commit()

    query = """
    CREATE TABLE IF NOT EXISTS data (
        id         INTEGER PRIMARY KEY,
        timestamp  REAL,
        open       REAL,
        high       REAL,
        low        REAL,
        close      REAL,
        volume     REAL
    )
    """
    cursor.execute(query)

    data = yf.download(ticker, period=period, interval=interval).dropna()
    for idx in data.index:
        ts = idx.timestamp()
        row = data.loc[idx]
        cursor.execute(
            """
        INSERT INTO data (timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
            (ts, row.Open[0], row.High[0], row.Low[0], row.Close[0], row.Volume[0]),
        )

    con.commit()
    con.close()


if __name__ == "__main__":
    download_example(
        db_path="data/examples/TQQQ.db",
        ticker="TQQQ",
        period="2y",
        interval="1h",
        flush=True,
    )
