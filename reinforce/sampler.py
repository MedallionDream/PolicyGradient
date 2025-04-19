import sqlite3
import numpy as np

from collections import deque
from typing import Tuple, Dict, List

from utils import Descriptors


class DataSampler:
    """
    A class for sampling data from a SQLite database for reinforcement learning tasks.

    This class handles data access, feature computation, and state construction for
    reinforcement learning environments using financial time series data.

    Attributes:
        _connection: SQLite database connection.
        _cursor: Database cursor for executing queries.
        _feature_params: Dictionary of parameters for feature extraction.
        _rows: Total number of rows in the database.
        _queue_size: Maximum size of the data queue.
        _queue: Deque storing the most recent data points.
        _counter: Current position in the dataset.
        _max_access: Maximum number of data points that can be accessed.
        _max_training_data: Maximum number of data points to use for training.
        _feature_funcs: Feature computation functions.

    Methods:
        counter: Property that returns the current position in the dataset.
        max_access: Property to get/set the maximum number of accessible data points.
        coef_of_var: Property that calculates the coefficient of variation of closing prices.
        reset: Resets the data queue and counter.
        sample_next: Samples the next data point and constructs the current state.
        _fetch_row: Fetches a specific row from the database.
        _construct_state: Constructs the state representation using feature functions.
    """

    def __init__(
        self,
        db_path: str,
        queue_size: int,
        feature_params: Dict[str, List[int] | Dict[str, List[int]]],
        max_access: int | None = None,
        max_training_data: int | None = None,
    ) -> None:

        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._cursor = self._connection.cursor()
        self._feature_params = feature_params

        self._cursor.execute("SELECT COUNT(*) FROM data")
        self._rows = self._cursor.fetchone()[0]
        self._queue_size = queue_size
        self._queue = deque(maxlen=queue_size)
        self._counter = 0

        if max_access is not None:
            max_access = min(self._rows - 2, max_access)

        if max_training_data is not None:
            if max_training_data < self._rows:
                self._counter = max(0, max_access - max_training_data)

        self._max_access = max_access
        self._max_training_data = max_training_data

        self._feature_funcs = Descriptors(feature_params)

    @property
    def counter(self) -> int:
        """
        Get the current counter value.

        Returns:
            int: The current count of trajectories or samples processed.
        """
        return self._counter

    @property
    def max_access(self) -> int:
        """
        Get the maximum number of accesses allowed.

        Returns:
            int: The maximum number of accesses allowed in the sampler.
        """
        return self._max_access

    @max_access.setter
    def max_access(self, m: int):
        """
        Set the maximum access limit for the sampler.

        This method adjusts the maximum number of rows that can be accessed from the buffer.

        Parameters
        ----------
        m : int
            The desired maximum access limit.

        Returns
        -------
        None

        Notes
        -----
        The method enforces bounds on the maximum access:
        - If m is greater than _rows - 2, it's set to _rows - 2
        - If m is less than _queue_size, it's set to _queue_size + 2
        - Otherwise, the value is used as provided
        """
        if m > self._rows - 2:
            m = self._rows - 2
        elif m < self._queue_size:
            m = self._queue_size + 2
        self._max_access = m

    @property
    def coef_of_var(self) -> float:
        """
        Calculate the coefficient of variation of the closing prices in the queue.

        The coefficient of variation is a measure of relative variability,
        calculated as the ratio of the standard deviation to the mean.

        Returns:
            float: The coefficient of variation of the closing prices in the queue.
        """
        data = np.asarray(list(self._queue))
        close = data[:, 4]
        cov = self._feature_funcs["cov"](close, len(self._queue))
        return cov[0]

    def reset(self) -> None:
        """
        Reset the sampler's state.

        This method reinitializes the internal queue and resets the counter based on the configured
        maximum training data limit. If a maximum training data limit is set and it's less than
        the number of rows, the counter is adjusted to respect this limit.

        Returns:
            None: This method doesn't return anything.
        """
        self._queue = deque(maxlen=self._queue_size)
        if self._max_training_data is not None:
            if self._max_training_data < self._rows:
                self._counter = max(0, self._max_access - self._max_training_data)
        else:
            self._counter = 0

    def sample_next(self) -> Tuple[bool, float, np.ndarray]:
        """
        Sample the next state from the underlying data source.

        Returns:
            Tuple:
            - bool: True if sampling is done (reached the end of data or max access limit), False otherwise
            - float: The reward value from the current row (at index 4)
            - np.ndarray: The constructed state representation

        Notes:
            - Increments the internal counter if not done
            - Uses _max_access limit if defined, otherwise uses data size (_rows - 2) as limit
            - Fetches the current row using _fetch_row and constructs state using _construct_state
        """
        if self._max_access is None:
            done = self._rows - 2 <= self._counter
        else:
            done = self._max_access <= self._counter

        row = self._fetch_row(self._counter)
        if not done:
            self._counter += 1

        return done, row[4], self._construct_state()

    def _fetch_row(self, index: int) -> Tuple[float]:
        """
        Fetches a single row from the SQLite database by its index.

        Args:
            index (int): The zero-based index of the row to fetch. Gets converted to 1-based for SQL query.

        Returns:
            Tuple[float]: The fetched row from the database.

        Note:
            The fetched row is also appended to the internal queue for caching purposes.
        """
        res = self._cursor.execute("SELECT * FROM data WHERE id = ?", (index + 1,))
        row = res.fetchone()
        self._queue.append(list(row))
        return row

    def _construct_state(self) -> np.ndarray:
        """
        Constructs the current state representation from the queue data.

        This method takes the content of the internal queue, converts it to a numpy array,
        and applies feature functions to compute a state representation that can be used
        by the learning algorithm.

        Returns:
            np.ndarray: A feature vector representing the current state, computed by
                        applying the feature functions to the queue data.
        """
        data = np.asarray(list(self._queue))
        return self._feature_funcs.compute(data)
