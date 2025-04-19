from typing import Dict
from collections import deque

from utils import Position, Trade, Action, Signal


class TradeManager:
    """
    Manages trading operations, positions, and portfolio tracking.

    The TradeManager class orchestrates trading decisions, maintains position states,
    tracks portfolio value, and logs trade history. It provides methods to attempt buying
    and selling based on trade signals, calculated risks, and position states.

    Attributes:
        _trade (Trade): Trade object that handles trade calculations and signals
        _position (Position): Current position state (Cash, Partial, or Asset)
        _full_port (bool): Flag indicating if full portfolio allocation is enabled
        _portfolio (float): Current portfolio value (starts at 1.0)
        _returns (list): List of trade returns
        _prev_exit (float): Price of the previous exit
        _trade_hist (deque): History of recent trades (max 10)

    Properties:
        cash (bool): Returns True if current position is Cash
        partial (bool): Returns True if current position is Partial
        asset (bool): Returns True if current position is Asset
        signal (Signal): Current trade signal
        portfolio (float): Current portfolio value
        returns (list): List of trade returns
        prev_exit (float): Price of previous exit
        curr_trade (Trade): Current trade object
        last_trade (Dict[str, float] | None): Details of the last executed trade

    Methods:
        log_trade(): Records current trade data in trade history
        try_buy(price, cov, amount=None): Attempts to buy based on given parameters
        try_sell(price, cov): Attempts to sell at the given price
        hold(cov): Updates risk without changing position
        potential_gain(price): Calculates potential gain at the given price
    """

    def __init__(
        self,
        cov: float,
        alpha: float,
        gamma: float,
        cost: float = 0,
        full_port: bool = False,
        leverage: float = 0,
    ) -> None:

        self._trade = Trade(cov, alpha, gamma, cost, full_port, leverage)
        self._position = Position.Cash
        self._full_port = full_port
        self._portfolio = 1.0
        self._returns = []
        self._prev_exit = 0.0
        self._trade_hist = deque(maxlen=10)

    @property
    def cash(self) -> bool:
        """
        Check if the current position is Cash.

        Returns:
            bool: True if the current position is Cash, False otherwise.
        """
        return self._position == Position.Cash

    @property
    def partial(self) -> bool:
        """
        Returns the partial position status of the manager.

        Returns:
            bool: True if the manager is in a partial position, False otherwise.
        """
        return self._position == Position.Partial

    @property
    def asset(self) -> bool:
        """
        A property that checks if the position is Asset.

        Returns:
            bool: True if the position is Asset, False otherwise.
        """
        return self._position == Position.Asset

    @property
    def signal(self) -> Signal:
        """
        Returns the trading signal from the trade.

        Returns
        -------
        Signal
            The signal from the trade, indicating the trading action
            to take (e.g., BUY, SELL, HOLD).
        """
        return self._trade.signal

    @property
    def portfolio(self):
        """
        Get the portfolio object.

        Returns:
            Portfolio: The portfolio object managed by this class.
        """
        return self._portfolio

    @property
    def returns(self):
        """
        Returns the stored return values.

        Returns
        -------
        list or None
            The list of return values if available, otherwise None.
        """
        return self._returns

    @property
    def prev_exit(self):
        """
        Returns the previous exit value.

        Returns:
            The value that was set as the previous exit, which can be of any type.
        """
        return self._prev_exit

    @property
    def curr_trade(self) -> Trade:
        """
        Get the current trade object.

        Returns:
            Trade: The current trade object being managed.
        """
        return self._trade

    @property
    def last_trade(self) -> Dict[str, float] | None:
        """
        Retrieves the most recent trade from the trade history.

        Returns:
            Dict[str, float] | None: A dictionary containing the details of the last trade,
                                    or None if no trades have been made.
        """
        return self._trade_hist[-1]

    def log_trade(self):
        """
        Log the current trade data to the trade history.

        This method appends the current trade's data to the internal trade history list.
        The trade data is accessed via the `self._trade.data` property.

        Returns:
            None
        """
        self._trade_hist.append(self._trade.data)

    def try_buy(self, price: float, cov: float, amount: float | None = None) -> Action:
        """
        Attempts to execute a buy action based on current trade conditions.

        This method checks if a trade is already open to determine if it should double down on a position.
        If no trade is open, it attempts to open a new trade.

        Args:
            price (float): The current price of the asset.
            cov (float): The covariance value, used to set the risk of the trade.
            amount (float | None, optional): The amount to buy. If None, a default amount will be used. Defaults to None.

        Returns:
            Action: The action that was taken. Possible values:
                - Action.Buy: A new position was opened.
                - Action.Double: An existing position was doubled.
                - Action.Sell: The position was sold.
                - Action.Hold: No action was taken.
        """
        self._trade.risk = cov
        if self._trade.opened:
            if self.partial and not self._full_port:
                # double down
                action = self._trade.double(price)
                if action == Action.Double:
                    self._position = Position.Asset
                    self._trade.hold()
                    return Action.Double
                elif action == Action.Sell:
                    self._portfolio = Position.Cash
                    self._portfolio = 0
                    self._trade.hold()
                    return Action.Sell
        else:
            # new trade
            success = self._trade.open(price, amount)
            if success:
                self._position = Position.Partial
                self._trade.hold()
                return Action.Buy
        return Action.Hold

    def try_sell(self, price: float, cov: float) -> float:
        """
        Attempts to sell an opened trade at the given price.

        Args:
            price (float): The price at which to attempt to sell.
            cov (float): The risk/covariance value to update in the trade before selling.

        Returns:
            float: The gain from the trade if it was successfully closed (positive value),
                or -1 if the trade was not opened and therefore couldn't be sold.

        Side effects:
            - Updates the trade's risk value
            - If the trade is successfully closed with a positive gain:
                - Updates the position to Cash
                - Multiplies the portfolio value by the gain
                - Appends the gain to the returns list
                - Updates the previous exit price
                - Logs the trade
                - Puts the trade on hold
        """
        self._trade.risk = cov
        if self._trade.opened:
            gain = self._trade.close(price)
            if gain > 0:
                self._position = Position.Cash
                self._portfolio *= gain
                self._returns += [gain]
                self._prev_exit = price
                self.log_trade()
                self._trade.hold()
            return gain
        return -1

    def hold(self, cov: float):
        """
        Modify the risk of the opened trade and put it in hold state.

        Args:
            cov (float): The covariance risk value to set for the trade.

        Notes:
            - This method sets the risk property of the trade to the provided covariance value.
            - If the trade is already opened, it calls the hold method on the trade.
            - No operation is performed if the trade is not opened.
        """
        self._trade.risk = cov
        if self._trade.opened:
            self._trade.hold()

    def potential_gain(self, price: float) -> float:
        """
        Calculate the potential gain at a given price.

        This method calculates the potential gain or loss that would be realized
        if the managed trade were to be closed at the specified price.

        Args:
            price (float): The hypothetical closing price to evaluate.

        Returns:
            float: The potential gain or loss amount. Positive values indicate profit,
                negative values indicate loss.
        """
        return self._trade.potential_gain(price)
