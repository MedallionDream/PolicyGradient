import time
import torch
import logging
import threading

from threading import Lock
from collections import deque
from typing import Dict, List, Optional

from backend import TradeManager, DataBuffer
from reinforce.model import PolicyNet, inference
from reinforce.environment import TradeEnv, train
from utils.trading import Position, Action, Status


class Server:
    """
    Server class to manage policy gradient reinforcement learning for stock trading.

    This class handles data buffering, model training, inference, and trade execution.
    It maintains state about the current trading position, model status, and provides
    methods for interaction with the trading environment.

    Parameters:
    ----------
    ticker : str
        The stock ticker symbol to trade
    period : str
        The historical data period to use (e.g., "1y", "max")
    interval : str
        The data interval (e.g., "1d", "1h", "15m")
    queue_size : int
        Size of the data buffer queue
    state_dim : int
        Dimension of the state representation
    action_dim : int
        Dimension of the action space
    embedding_dim : int
        Size of the embedding dimension for the policy network
    inaction_cost : float
        Cost penalty for holding a position
    action_cost : float
        Transaction cost for taking buy/sell actions
    device : str
        Device to run computations on ("cpu" or "cuda")
    return_thresh : float
        Return threshold for considering successful trades
    retrain_freq : int
        Frequency of model retraining in timer ticks
    training_params : Dict[str, int | float]
        Dictionary of training hyperparameters
    feature_params : Dict[str, List[int] | Dict[str, List[int]]]
        Parameters for feature engineering
    db_path : Optional[str], default=None
        Path to database file, defaults to "../data/{ticker}.db"
    live_data : bool, default=False
        Whether to use live market data
    sharpe_cutoff : int, default=30
        Minimum number of samples to calculate Sharpe ratio
    full_port : bool, default=False
        Whether to use full portfolio for trades
    gamma : float, default=0.1
        Discount factor for future rewards
    alpha : float, default=1.5
        Risk-aversion parameter
    beta : float, default=0.5
        Secondary risk parameter
    zeta : float, default=0.5
        Parameter for reward calculation
    leverage : float, default=1.0
        Trading leverage multiplier
    num_mem : int, default=512
        Size of memory buffer
    mem_dim : int, default=128
        Dimension of memory entries
    inference_method : str, default="prob"
        Method used for inference ("prob" for probabilistic)
    checkpoint_path : str, default="checkpoint"
        Path to save model checkpoints
    logger : Optional[logging.Logger], default=None
        Logger instance
    max_training_data : int | None, default=None
        Maximum amount of data points to use for training

    Methods:
    -------
    save_model(name: str) -> None
        Saves the current model to disk with the specified name
    load_model(path: str) -> None
        Loads model weights from the specified path
    status_report() -> Dict[str, int | bool | str]
        Returns current status information
    consume_queue() -> List[Dict[str, int | float | str]] | None
        Retrieves and clears the data queue
    start_timer(interval: int) -> None
        Starts the timer thread that handles scheduled operations
    tohlcv() -> Dict[str, float]
        Returns the latest OHLCV data
    fetch_buffer() -> Dict[str, Dict[str, float]]
        Returns the entire data buffer
    update_model() -> bool
        Updates the model with latest weights
    train_model(...) -> None
        Begins asynchronous model training
    join_timer_thread() -> bool
        Waits for the timer thread to complete
    join_train_thread() -> bool
        Waits for the training thread to complete
    join_inference_thread() -> bool
        Waits for the inference thread to complete
    """

    def __init__(
        self,
        ticker: str,
        period: str,
        interval: str,
        queue_size: int,
        state_dim: int,
        action_dim: int,
        embedding_dim: int,
        inaction_cost: float,
        action_cost: float,
        device: str,
        return_thresh: float,
        retrain_freq: int,
        training_params: Dict[str, int | float],
        feature_params: Dict[str, List[int] | Dict[str, List[int]]],
        db_path: Optional[str] = None,
        live_data: bool = False,
        sharpe_cutoff: int = 30,
        full_port: bool = False,
        gamma: float = 0.1,
        alpha: float = 1.5,
        beta: float = 0.5,
        zeta: float = 0.5,
        leverage: float = 1.0,
        num_mem: int = 512,
        mem_dim: int = 128,
        inference_method: str = "prob",
        checkpoint_path: str = "checkpoint",
        logger: Optional[logging.Logger] = None,
        max_training_data: int | None = None,
    ) -> None:

        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)

        self._ticker = ticker
        self._period = period
        self._interval = interval
        self._live_data = live_data
        self._new_session = True
        self._mutex = Lock()
        self._position = Position.Cash
        self._device = device
        self._inference_method = inference_method

        if not db_path:
            db_path = f"../data/{ticker}.db"

        self._buffer = DataBuffer(
            ticker=ticker,
            period=period,
            interval=interval,
            queue_size=queue_size,
            db_path=db_path,
            feature_params=feature_params,
            logger=logger,
            live_data=live_data,
        )

        if live_data:
            self._buffer.write_queue_to_db(flush=True)

        self._model = PolicyNet(
            input_dim=state_dim,
            output_dim=action_dim,
            position_dim=len(Position),
            embedding_dim=embedding_dim,
            num_mem=num_mem,
            mem_dim=mem_dim,
        ).to(device)

        self._env = TradeEnv(
            state_dim=state_dim,
            action_dim=action_dim,
            embedding_dim=embedding_dim,
            queue_size=queue_size,
            inaction_cost=inaction_cost,
            action_cost=action_cost,
            device=device,
            db_path=db_path,
            sharpe_cutoff=sharpe_cutoff,
            return_thresh=return_thresh,
            testing=not live_data,
            max_training_data=max_training_data,
            feature_params=feature_params,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            zeta=zeta,
            leverage=leverage,
            mem_dim=mem_dim,
            num_mem=num_mem,
        )

        self._manager = TradeManager(
            cov=self._buffer.coef_of_var,
            alpha=alpha,
            gamma=gamma,
            cost=action_cost,
            full_port=full_port,
        )

        self._nounce = 0
        self._beat = 0
        self._status = Status(gamma * self._buffer.coef_of_var, alpha)
        self._max_access_accum = 0
        self._checkpoint_path = checkpoint_path
        self._training_params = training_params
        self._ready = False
        self._training = False
        self._train_thread = None
        self._inferencing = False
        self._inference_thread = None
        self._timer_thread = None
        self._train_counter = retrain_freq
        self._retrain_freq = retrain_freq
        self._terminate = False
        self._gamma = gamma
        self._epsilon = 0.9
        self._epsilon_decay = 0.9
        self._data_queue = deque(maxlen=25)

    def __del__(self):
        """
        Destructor method for the class.

        This method ensures proper cleanup by joining any active timer, inference, or training threads
        before the instance is garbage collected. It prevents thread leaks by:
        1. Always joining the timer thread
        2. Joining the inference thread if inference was in progress
        3. Joining the training thread if training was in progress

        Note: This is automatically called by Python's garbage collector when the instance is about to be destroyed.
        """
        self.join_timer_thread()

        if self._inferencing:
            self.join_inference_thread()

        if self._training:
            self.join_train_thread()

    @property
    def new_session(self) -> bool:
        """
        Checks if this is a new session and resets the flag.

        Returns:
            bool: True if this is a new session, False otherwise. If True,
                the new session flag is reset to False.
        """
        if self._new_session:
            self._new_session = False
            return True
        return False

    @property
    def session_info(self) -> Dict[str, str]:
        """
        Returns information about the current trading session.

        This method provides details about the session configuration including
        the stock ticker, time period, and interval for data analysis.

        Returns:
            Dict[str, str]: A dictionary containing:
                - "type": Always "new_session" indicating a session info message
                - "ticker": The stock ticker symbol being analyzed
                - "period": The time period for historical data, or "live" if using live data
                - "interval": The time interval between data points
        """
        return {
            "type": "new_session",
            "ticker": self._ticker,
            "period": self._period if not self._live_data else "live",
            "interval": self._interval,
        }

    @property
    def busy(self):
        """
        Check if the server is currently in training mode.

        Returns:
            bool: True if the server is currently training, False otherwise.
        """
        return self._training

    @property
    def inf_busy(self):
        """
        Check if the model is currently being used for inferencing.

        Returns:
            bool: True if inferencing is in progress, False otherwise.
        """
        return self._inferencing

    def _append_action(
        self, timestamp: str, action: Action, price: float, prob: float, amount: float
    ) -> None:
        """
        Append action information to the data queue.

        Args:
            timestamp (str): The timestamp when the action was taken.
            action (Action): The trading action performed (e.g., BUY, SELL).
            price (float): The price at which the action was executed.
            prob (float): The probability or confidence score associated with the action.
            amount (float): The quantity or amount involved in the action.

        Note:
            This method formats the action data as a dictionary and adds it to the data queue
            for later processing or transmission.
        """
        self._data_queue.append(
            {
                "type": "action",
                "timestamp": timestamp,
                "action": action.name,
                "close": price,
                "probability": prob,
                "amount": amount,
            }
        )

    def _append_trade(self, timestamp: str, entry: float, exit: float, amount: float):
        """
        Append a trade record to the data queue.

        Args:
            timestamp (str): The timestamp of the trade.
            entry (float): The entry price of the trade.
            exit (float): The exit price of the trade.
            amount (float): The amount of the trade.

        Returns:
            None

        Note:
            This is an internal method used to track trade data for visualization or analysis.
        """
        self._data_queue.append(
            {
                "type": "trade",
                "timestamp": timestamp,
                "entry": entry,
                "exit": exit,
                "amount": amount,
            }
        )

    def save_model(self, name: str) -> None:
        """
        Saves the current model weights to a file.

        The method obtains the model weights from the environment and saves them to a file
        at the specified path using PyTorch's save functionality. Access to the model weights
        is protected by a mutex to ensure thread safety.

        Args:
            name (str): The name of the file to save the model weights to.
                        This will be appended to the checkpoint path.

        Returns:
            None

        Note:
            The method uses a mutex to ensure thread-safe access to the model weights.
        """
        path = self._checkpoint_path + "/" + name
        with self._mutex:
            state_dict = self._env.model_weights
        torch.save(state_dict, path)

    def load_model(self, path: str) -> None:
        """
        Load a pre-trained model from a file.

        This method loads a PyTorch model's state dictionary from the specified path and
        updates the current model. The operation is thread-safe, using a mutex to prevent
        concurrent access issues.

        Args:
            path (str): The file path where the model's state dictionary is stored.

        Returns:
            None

        Raises:
            FileNotFoundError: If the specified path does not exist.
            RuntimeError: If the model cannot be loaded properly.
        """
        state_dict = torch.load(path)
        with self._mutex:
            self._env.model = state_dict

    def status_report(self) -> Dict[str, int | bool | str]:
        """
        Generate a status report dictionary for the current state of the system.

        Returns:
            Dict[str, int | bool | str]: A dictionary containing the following keys:
                - "type": Always set to "report".
                - "training": Boolean indicating whether training is in progress.
                - "ready": Boolean indicating whether the system is ready.
                - "done": Boolean indicating whether the buffer processing is complete.
        """
        r = {
            "type": "report",
            "training": self._training,
            "ready": self._ready,
            "done": self._buffer.done,
        }
        self._data_queue.append(r)
        return r

    def consume_queue(self) -> List[Dict[str, int | float | str]] | None:
        """
        Consumes and returns all data from the queue.

        This method safely retrieves all data currently in the queue and clears the queue.
        Thread safety is ensured using a mutex lock.

        Returns:
            List[Dict[str, int | float | str]] | None: A list of all dictionaries in the queue if the queue
            is not empty, otherwise None. Each dictionary contains data with string keys and values
            that may be integers, floats, or strings.
        """
        with self._mutex:
            if len(self._data_queue) > 0:
                d = list(self._data_queue)
                self._data_queue.clear()
                return d
        return None

    def _timer_loop(self, interval: int):
        """
        Starts a timer loop that periodically performs model training, updates OHLC data, and runs inference.

        The loop first initializes by training the model with the configured parameters, then enters a cycle
        where it:
        1. Generates and queues OHLC data
        2. Periodically retrains the model based on the retrain frequency counter
        3. Updates the buffer queue
        4. Runs inference if the model is ready
        5. Sleeps for the specified interval

        The loop continues until either self._terminate is set to True or the buffer is done.

        Args:
            interval (int): The time interval in seconds between loop iterations

        Returns:
            None
        """
        self._logger.info("starting timer thread")
        self.train_model(
            episodes=self._training_params["episodes"],
            learning_rate=self._training_params["learning_rate"],
            momentum=self._training_params["momentum"],
            weight_decay=self._training_params["weight_decay"],
            max_grad_norm=self._training_params["max_grad_norm"],
            min_episodes=self._training_params["min_episodes"],
        )

        while not self._terminate and not self._buffer.done:
            ohlc = self.tohlcv()
            ohlc["type"] = "ohlc"
            ohlc["nounce"] = self._nounce
            self._nounce += 1
            self._data_queue.append(ohlc)
            if self._train_counter == 0:
                self._logger.info("running scheduled training")
                self.train_model(
                    episodes=self._training_params["episodes"],
                    learning_rate=self._training_params["learning_rate"],
                    momentum=self._training_params["momentum"],
                    weight_decay=self._training_params["weight_decay"],
                    max_grad_norm=self._training_params["max_grad_norm"],
                    min_episodes=self._training_params["min_episodes"],
                )
                self._train_counter = self._retrain_freq

            self._buffer.update_queue(False)
            self._logger.info("running scheduled inference")
            if self._ready:
                self._inference()
            else:
                self._logger.warning("model not ready, not running inference")
            self._train_counter -= 1
            time.sleep(interval)
        self._logger.warning("timer loop terminated")

    def start_timer(self, interval: int):
        """
        Start a timer thread that executes a function at regular intervals.

        This method creates and starts a new thread that will call the `_timer_loop`
        method repeatedly at the specified interval. The thread reference is stored
        for future management.

        Parameters:
            interval (int): The time in seconds between each execution of the timer function.

        Returns:
            None

        Note:
            The timer thread runs until explicitly stopped. Access to the thread reference
            is protected by a mutex to ensure thread safety.
        """
        thread = threading.Thread(target=self._timer_loop, args=(interval,))
        thread.start()

        with self._mutex:
            self._timer_thread = thread

    def tohlcv(self) -> Dict[str, float]:
        """
        Returns the latest TOHLCV data from the buffer (Time, Open, High, Low, Close, Volume).
        This method retrieves the last data point from the buffer and updates the maximum access count.
        If the server is not in training mode, it increments the maximum access count for the sampler.
        If the server is in training mode, it accumulates the access count for later use.
        Returns:
            Dict[str, float]: A dictionary containing the latest OHLCV data.
            The keys are 'timestamp', 'open', 'high', 'low', 'close', and 'volume'.
        Note:
            The method is thread-safe and uses a mutex lock to prevent concurrent access issues.
        """
        if not self._training:
            if self._max_access_accum > 0:
                self._env.sampler.max_access += self._max_access_accum + 1
                self._max_access_accum = 0
            else:
                self._env.sampler.max_access += 1
        else:
            self._max_access_accum += 1
        return self._buffer.last_tohlcv()

    def fetch_buffer(self) -> Dict[str, Dict[str, float]]:
        """
        Returns a dictionary representation of the current buffered PG timesteps.

        The method acquires a lock to ensure thread safety when accessing the buffer,
        then returns the current queue of the buffer as a dictionary mapping state keys
        to dictionaries of action probabilities.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary where each key is a state representation
                                        and each value is a dictionary mapping actions to their
                                        probability values.
        """
        with self._mutex:
            return self._buffer.queue

    def update_model(self) -> bool:
        """
        Update the model weights with those from the environment.

        This method acquires a mutex lock to safely update the model's weights.
        It only performs the update if the model is not currently training
        or being used for inference.

        Returns:
            bool: True if the model was successfully updated, False otherwise.
        """
        self._mutex.acquire_lock()
        if not self._training and not self._inferencing:
            self._model.load_state_dict(self._env.model_weights)
            self._mutex.release_lock()
            self._logger.info("model updated successfully")
            return True
        else:
            self._logger.warning("model currently being trained, cannot copy weights")
            self._mutex.release_lock()
            return False

    def _train(
        self,
        episodes: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        max_grad_norm: float,
        min_episodes: int,
    ) -> None:
        """
        Trains the perceptron model with the given hyperparameters.

        This method manages the training process, maintaining thread safety through mutex locks.
        It calls the training function with the specified parameters, and updates the model if
        the performance exceeds the previous best.

        Args:
            episodes (int): The number of episodes to train for.
            learning_rate (float): The learning rate for the optimizer.
            momentum (float): The momentum for the optimizer.
            weight_decay (float): The weight decay (L2 penalty) for the optimizer.
            max_grad_norm (float): The maximum norm of the gradients for gradient clipping.
            min_episodes (int): The minimum number of episodes required before early stopping.

        Returns:
            None: This method doesn't return anything but updates the internal model state.

        Note:
            This method is thread-safe due to the use of mutex locks for the training state.
            The model is only updated if the new performance (beat) exceeds the previous best.
        """
        with self._mutex:
            self._training = True

        beat = train(
            env=self._env,
            episodes=episodes,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            min_episodes=min_episodes,
        )

        with self._mutex:
            self._training = False

        if beat > self._beat:  # or self._beat == 0:
            # if self._epsilon < random.random():
            # self._epsilon *= self._epsilon_decay
            self._beat = beat
            self.update_model()

        with self._mutex:
            if not self._ready:
                self._ready = True

    def train_model(
        self,
        episodes: int,
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.9,
        max_grad_norm: float = 1.0,
        min_episodes: int = 10,
    ):
        """
        Start training the model in a separate thread.

        This method initiates the training process in a new thread if no other training
        is currently in progress. It prevents concurrent training operations by checking
        the training state with a mutex lock.

        Parameters
        ----------
        episodes : int
            The number of episodes to train for
        learning_rate : float, optional
            Learning rate for optimization, by default 1e-3
        momentum : float, optional
            Momentum factor for optimization, by default 0.9
        weight_decay : float, optional
            Weight decay (L2 penalty) factor, by default 0.9
        max_grad_norm : float, optional
            Maximum norm of the gradients for gradient clipping, by default 1.0
        min_episodes : int, optional
            Minimum number of episodes to train for, by default 10

        Returns
        -------
        None
            This method doesn't return any value but starts a training thread if none is running

        Notes
        -----
        The actual training occurs in the `_train` method, which is executed in a separate thread.
        The method logs when training starts, including the thread ID.
        """
        with self._mutex:
            if self._training:
                self._logger.info(
                    f"there is another training thread running, wait for it to finish first"
                )
                return

        thread = threading.Thread(
            target=self._train,
            args=(
                episodes,
                learning_rate,
                momentum,
                weight_decay,
                max_grad_norm,
                min_episodes,
            ),
        )

        thread.start()
        self._logger.info(f"training started, thread id: {thread}")

        with self._mutex:
            self._train_thread = thread

    def join_timer_thread(self) -> bool:
        """
        Join the timer thread, if it exists.

        This method waits for the timer thread to complete its execution.
        If there is no timer thread running, a warning is logged.

        Returns:
            bool: True if the timer thread was joined successfully, False if no timer thread exists.

        Thread safety:
            This method is thread-safe, using self._mutex to protect access to self._timer_thread.
        """
        with self._mutex:
            if not self._timer_thread:
                self._logger.warning("no running timer thread, nothing to end...")
                return False

        self._timer_thread.join()
        return True

    def join_train_thread(self) -> bool:
        """
        Joins the training thread if one is active.

        This method waits for the training thread to complete execution. It does nothing if no
        training is currently in progress.

        Returns:
            bool: True if a training thread was joined successfully, False if no training was in progress.
        """
        with self._mutex:
            if not self._training:
                self._logger.warning("model not training, nothing to end...")
                return False

        self._train_thread.join()
        return True

    def join_inference_thread(self) -> bool:
        """
        Join the inference thread if it is running.

        This method joins the inference thread, which causes the calling thread
        to wait until the inference thread has finished execution. This is typically
        used to ensure that the inference thread has completed before continuing.

        Returns:
            bool: True if the inference thread was joined successfully, False if
                there was no inference thread running.
        """
        with self._mutex:
            if not self._inferencing:
                self._logger.warning("model not inferencing, nothing to end...")
                return False

        self._inference_thread.join()
        return True

    def _inference(self) -> None:
        """
        Initiates an inference operation in a separate thread.

        This method ensures that only one inference operation can run at a time.
        If another inference thread is already running, it logs the information
        and returns without starting a new thread.

        Returns:
            None
        """
        with self._mutex:
            if self._inferencing:
                self._logger.info(
                    f"there is another inferencing thread running, wait for it to finish first"
                )
                return

        thread = threading.Thread(
            target=self._void_inference,
            args=(),
        )

        thread.start()
        with self._mutex:
            self._inference_thread = thread

    def _void_inference(self) -> None:
        """
        Execute model inference to determine trading actions.

        This method fetches the current state from the buffer, runs inference on the model
        to determine the optimal action (Buy, Sell, or Hold), and executes the action
        through the manager. The method implements mutual exclusion to prevent concurrent
        inference runs.

        The method checks for model readiness and data availability before proceeding.
        If another inference thread is running, it logs a warning and returns without
        performing inference.

        Trading decisions are based on:
        - Current model prediction
        - Probability of the prediction
        - Coefficient of variation from the buffer
        - Potential gain calculations

        When actions are taken, they are recorded along with relevant trade information.

        Returns:
            None
        """
        with self._mutex:
            if not self._ready:
                self._logger.warning("model not ready, not running inference")
                return

        state = self._buffer.fetch_state(False)
        data = self._buffer.queue["data"][-1]
        price = data["close"]
        if len(state) == 0:
            self._logger.warning("no data available, not running inference")
            self._ready = False
            return None

        with self._mutex:
            if self._inferencing:
                self._logger.info(
                    f"there is another inference thread running, wait for it to finish first"
                )
                return

            self._inferencing = True

        result, prob = inference(
            model=self._model,
            state=state,
            position=int(self._position.value),
            potential=self._manager.potential_gain(price) - 1,
            device=self._device,
            method=self._inference_method,
        )

        action = Action(result)
        actual_action = Action.Hold
        with self._mutex:
            self._risk_free_rate = price / self._buffer.queue["data"][0]["close"]
            # Validate buy
            if action == Action.Buy:
                actual_action = self._manager.try_buy(
                    price, self._buffer.coef_of_var, prob
                )
            elif action == Action.Sell:
                gain = self._manager.try_sell(price, self._buffer.coef_of_var)
                if gain >= 0:
                    actual_action = Action.Sell
                    trade = self._manager.last_trade
                    self._append_trade(
                        data["timestamp"],
                        trade["entry"],
                        trade["exit"],
                        trade["amount"],
                    )

            if action != Action.Hold:
                self._append_action(
                    data["timestamp"],
                    actual_action,
                    price,
                    prob,
                    self._manager.curr_trade.amount,
                )
        self._inferencing = False
