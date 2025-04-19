import logging
import sqlite3
import numpy as np
import yfinance as yf

from collections import deque
from typing import Dict, List, Optional

from utils import Descriptors


class DataBuffer:
    """
    DataBuffer manages financial data retrieval and manipulation for both historical and live market data.

    This class provides a buffer for storing and accessing OHLC (Open, High, Low, Close) and volume data
    for a specified ticker, either from a local SQLite database or live from Yahoo Finance.
    It supports feature calculation through the Descriptors class and maintains a fixed-size queue
    of the most recent data points.

    Attributes:
        _logger (logging.Logger): Logger for recording events and errors.
        _queue_size (int): Maximum number of data points to store in the queue.
        _live_data (bool): Flag indicating whether to use live data from Yahoo Finance.
        _ticker (str): The stock ticker symbol.
        _period (str): Time period to download data for (e.g., '1d', '1mo').
        _interval (str): Time interval between data points (e.g., '1m', '1h', '1d').
        _queue (collections.deque): Double-ended queue holding the data points.
        _db_path (str): Path to the SQLite database file.
        _feature_params (Dict): Parameters for calculating technical indicators.
        _done (bool): Flag indicating whether all data has been processed.
        _con (sqlite3.Connection): Connection to SQLite database.
        _cursor (sqlite3.Cursor): Cursor for database operations.
        _counter (int): Counter for tracking position in database.
        _rows (int): Total number of rows in the database.
        _feature_funcs (Descriptors): Object for calculating technical indicators.

    Methods:
        reset(): Resets the buffer to its initial state.
        update_queue(write_to_db): Updates the queue with the next data point.
        last_tohlcv(): Returns the last data point in TOHLCV format.
        fetch_state(update): Constructs and returns the current state.
        write_queue_to_db(flush): Writes the current queue to the database.
    """

    def __init__(
        self,
        ticker: str,
        period: str,
        interval: str,
        queue_size: int,
        db_path: str,
        feature_params: Dict[str, List[int] | Dict[str, List[int]]],
        logger: Optional[logging.Logger] = None,
        live_data: bool = False,
    ) -> None:

        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)

        self._queue_size = queue_size
        self._live_data = live_data
        self._ticker = ticker
        self._period = period
        self._interval = interval
        self._queue = deque(maxlen=queue_size)
        self._db_path = db_path
        self._feature_params = feature_params
        self._done = False

        self._con = None
        self._cursor = None
        self._counter = queue_size + 2
        self._rows = 0

        if not live_data:
            self._logger.info(f"not live data, connecting to db {self._db_path}")
            self._con = sqlite3.connect(db_path, check_same_thread=False)
            self._cursor = self._con.cursor()
            self._cursor.execute("SELECT COUNT(*) FROM data")
            self._rows = self._cursor.fetchone()[0]

        self._feature_funcs = Descriptors(feature_params)
        self._fill_queue()

    def reset(self) -> None:
        """
        Reset the buffer to its initial state.

        If the buffer is a live buffer (i.e., `_live_data` is True), it cannot be reset, and a warning will be logged.
        Otherwise, it:
        - Sets the done flag to False
        - Resets the counter to queue_size + 2
        - Creates a new deque with the specified maximum length
        - Calls _fill_queue() to populate the buffer

        Returns:
            None
        """
        if self._live_data:
            logging.warning("live buffer cannot be reset")
            return

        self._done = False
        self._counter = self._queue_size + 2
        self._queue = deque(maxlen=self._queue_size)
        self._fill_queue()

    @property
    def coef_of_var(self) -> float:
        """
        Calculate the coefficient of variation for the close prices in the queue.

        The coefficient of variation is a standardized measure of dispersion that
        represents the ratio of the standard deviation to the mean, expressed as a percentage.

        Returns:
            float: The coefficient of variation value for the close prices.
        """
        data = np.asarray(list(self._queue))
        close = data[:, 4]
        cov = self._feature_funcs["cov"](close, len(self._queue))
        return cov[0]

    @property
    def done(self) -> bool:
        """
        Check if the buffer is in a "done" state.

        Returns:
            bool: True if the buffer is done, False otherwise.
        """
        return self._done

    @property
    def queue(self) -> Dict[str, Dict[str, float]]:
        """
        Returns the queue as a dictionary with formatted data.

        The function converts the internal queue representation into a dictionary format
        with a "data" key containing a list of dictionaries. Each dictionary in the list
        represents a row from the queue with keys for "timestamp", "open", "high", "low",
        "close", and "volume".

        Returns:
            Dict[str, Dict[str, float]]: A dictionary with a "data" key containing a list of
            dictionaries, each representing a row in the queue with OHLCV data.
        """
        data = {"data": []}
        for row in self._queue:
            data["data"] += [
                {
                    "timestamp": row[0],
                    "open": row[1],
                    "high": row[2],
                    "low": row[3],
                    "close": row[4],
                    "volume": row[5],
                }
            ]
        return data

    def _fill_queue(self) -> None:
        """
        Fill the internal queue with historical or mock data.

        If `_live_data` is True, fetches historical data for the configured ticker, period, and interval
        using yfinance.download() and populates the queue with this data.

        If `_live_data` is False, generates mock data by calling update_queue() multiple times until
        the queue has sufficient data points (2 * queue_size + 2).

        The data in the queue follows the format: [timestamp, Open, High, Low, Close, Volume].

        Returns:
            None
        """
        if self._live_data:
            data = yf.download(
                tickers=self._ticker, period=self._period, interval=self._interval
            )
        else:
            while self._counter < self._queue_size * 2 + 2:
                self.update_queue(False)
            return

        for idx in data.index:
            row = data.loc[idx]
            self._queue.append(
                [idx.timestamp(), row.Open, row.High, row.Low, row.Close, row.Volume]
            )

    def update_queue(self, write_to_db: bool = False) -> None:
        """
        Updates the queue with new data based on the mode of operation.

        If not in live data mode and a cursor is available, fetches data from the database
        at the next sequential ID and adds it to the queue. If the cursor has reached near
        the end of available rows, marks the process as done.

        If in live data mode, gets the last TOHLCV (Time, Open, High, Low, Close, Volume) data
        and adds it to the queue. Optionally writes this data to the database.

        Args:
            write_to_db (bool, optional): Whether to write the latest data to the database
                                        when in live data mode. Defaults to False.

        Returns:
            None: This method doesn't return a value but updates the internal queue.
        """
        if not self._live_data:
            if self._cursor:
                if self._counter >= self._rows - 2:
                    self._done = True
                    return

                res = self._cursor.execute(
                    "SELECT * FROM data WHERE id = ?", (self._counter + 1,)
                )
                row = res.fetchone()
                self._queue.append(list(row)[1:])
                self._counter += 1
                return
            else:
                self._logger.error("no connection to database")
                return

        data = self.last_tohlcv()
        self._queue.append(list(data.values()))

        if write_to_db:
            self.write_last_row_to_db()

    def last_tohlcv(self) -> Dict[str, float]:
        """
        Retrieve the most recent TOHLCV (timestamp, open, high, low, close, volume) data.

        This method fetches the latest price and volume data for the configured ticker symbol.
        If live data mode is enabled, it downloads the data using Yahoo Finance.
        Otherwise, it retrieves the last entry from the internal queue.

        Returns:
            Dict[str, float]: A dictionary containing the timestamp, open, high, low, close,
            and volume values of the most recent data point. Returns an empty dictionary
            if there's no connection to the database and no cursor is available.

        Raises:
            No explicit exceptions raised, but logs an error if no connection to database.
        """
        if self._live_data:
            data = yf.download(
                tickers=self._ticker, period="1d", interval=self._interval
            )

        else:
            # self.update_queue(False)
            if self._cursor:
                item = self._queue[-1]
                return {
                    "timestamp": item[0],
                    "open": item[1],
                    "high": item[2],
                    "low": item[3],
                    "close": item[4],
                    "volume": item[5],
                }
            else:
                self._logger.error("no connection to database")
                return {}

        last = data.loc[data.index[-1]]
        return {
            "timestamp": data.index[-1].timestamp(),
            "open": last.Open,
            "high": last.High,
            "low": last.Low,
            "close": last.Close,
            "volume": last.Volume,
        }

    def fetch_state(self, update: bool = True) -> np.ndarray:
        """
        Fetch the current state of the buffer.

        This method returns the current state of the buffer by calling the `_construct_state` method.
        If `update` is True (default), it will first update the queue using the `update_queue` method.

        Parameters
        ----------
        update : bool, optional
            If True, update the queue before constructing the state. Default is True.

        Returns
        -------
        np.ndarray
            The constructed state representation.
        """
        if update:
            self.update_queue()

        return self._construct_state()

    def _construct_state(self) -> np.ndarray:
        """
        Constructs the state representation from the current queue.

        This method converts the queue data to a numpy array and applies the
        feature functions to compute the state representation.

        Returns:
            np.ndarray: The computed state representation based on the current queue data.
        """
        data = np.asarray(list(self._queue))
        return self._feature_funcs.compute(data)

    def _write_last_row_to_db(self) -> None:
        """
        Writes the last row from the internal queue to the SQLite database.

        This method:
            1. Connects to the SQLite database at the path stored in self._db_path
            2. Gets the last row from the internal queue (self._queue)
            3. Inserts the row into the 'data' table with columns: timestamp, open, high, low, close, volume
            4. Commits the transaction and closes the database connection

        Note: The connection is created with check_same_thread=True to allow access from any thread.
        """
        con = sqlite3.connect(self._db_path, check_same_thread=True)
        cursor = con.cursor()

        row = self._queue[-1]
        cursor.execute(
            """
            INSERT INTO data (timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (row[0], row[1], row[2], row[3], row[4], row[5]),
        )

        con.commit()
        con.close()

    def write_queue_to_db(self, flush: bool = True) -> None:
        """
        Write data from the internal queue to SQLite database.

        This method connects to the SQLite database and writes all rows from the
        internal queue to a 'data' table. If flush is True, all existing tables
        are dropped and a new 'data' table is created with the schema for OHLCV data.

        Parameters:
        -----------
        flush : bool, default=True
            If True, all existing tables in the database will be dropped and
            recreated. If False, data will be appended to existing tables.

        Notes:
        ------
        - The data table schema includes: id, timestamp, open, high, low, close, volume
        - After writing data, the database connection is committed and closed
        - This method does not clear the internal queue after writing
        """
        con = sqlite3.connect(self._db_path, check_same_thread=True)
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
                self._logger.info(f"Table {table_name} dropped successfully")

            self._logger.info("All user tables dropped successfully")

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

        for row in self._queue:
            cursor.execute(
                """
            INSERT INTO data (timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
                (row[0], row[1], row[2], row[3], row[4], row[5]),
            )

        con.commit()
        con.close()
