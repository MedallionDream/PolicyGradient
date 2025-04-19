import logging
import numpy as np
import gymnasium as gym
import torch.optim as optim
import torch.nn as nn

from gymnasium import spaces
from typing import Dict, List, Any, Optional

from backend.manager import TradeManager
from reinforce.sampler import DataSampler
from reinforce.model import PolicyNet, select_action, compute_loss
from utils.descriptors import compute_sharpe_ratio
from utils.trading import Action, Position


class TradeEnv(gym.Env):
    """
    A trading environment class that implements OpenAI Gym's interface for reinforcement learning.
    This class simulates a trading environment where an agent can take actions like buying, selling,
    or holding assets. It tracks portfolio performance, manages trading decisions, and calculates
    rewards based on trading outcomes.
    Attributes:
        action_space (spaces.Discrete): The action space defining possible trading actions.
        observation_space (spaces.Discrete): The observation space defining the state representation.
    Parameters:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space (number of possible actions).
        embedding_dim (int): Dimension of the embeddings used in the policy network.
        queue_size (int): Size of the data queue for sampling.
        inaction_cost (float): Cost penalty for holding/inaction.
        action_cost (float): Transaction cost for taking buy/sell actions.
        device (str): Device to run the model on ('cpu' or 'cuda').
        db_path (str): Path to the database containing trading data.
        return_thresh (float): Threshold for portfolio returns to terminate an episode.
        sharpe_cutoff (int, optional): Number of returns to consider for Sharpe ratio calculation. Defaults to 30.
        alpha (float, optional): Risk aversion parameter. Defaults to 1.5.
        beta (float, optional): Secondary risk parameter. Defaults to 0.5.
        gamma (float, optional): Discount factor for future rewards. Defaults to 0.15.
        zeta (float, optional): Balance parameter between Sharpe ratio and returns in reward calculation. Defaults to 0.5.
        num_mem (int, optional): Number of memory units in the policy network. Defaults to 512.
        mem_dim (int, optional): Dimension of each memory unit. Defaults to 256.
        feature_params (Dict[str, List[int] | Dict[str, List[int]]], optional): Parameters for feature extraction. Defaults to None.
        logger (Optional[logging.Logger], optional): Logger instance. Defaults to None.
        testing (bool, optional): Whether in testing mode. Defaults to False.
        leverage (float, optional): Leverage allowed in trading. Defaults to 0.
        max_training_data (int | None, optional): Maximum amount of training data to use. Defaults to None.
    Properties:
        beta: Returns the beta parameter.
        sampler: Returns the data sampler instance.
        model: Returns the policy network model.
        model_weights: Gets or sets the model's state dictionary.
        log_return: Returns the cumulative log return.
        log_probs: Returns the log probabilities of taken actions.
        portfolio: Returns the current portfolio value.
        init_close: Returns the initial closing price.
    Methods:
        reset(): Resets the environment to initial state and returns the first observation.
        step(action): Takes an action in the environment and returns the next state, reward, done flags, and info.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embedding_dim: int,
        queue_size: int,
        inaction_cost: float,
        action_cost: float,
        device: str,
        db_path: str,
        return_thresh: float,
        sharpe_cutoff: int = 30,
        alpha: float = 1.5,
        beta: float | None = 0.5,
        gamma: float = 0.15,
        zeta: float = 0.5,
        num_mem: int = 512,
        mem_dim: int = 256,
        feature_params: Dict[str, List[int] | Dict[str, List[int]]] | None = None,
        logger: Optional[logging.Logger] = None,
        testing: bool = False,
        leverage: float = 0,
        max_training_data: int | None = None,
    ) -> None:

        super().__init__()

        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)

        self.action_space = spaces.Discrete(action_dim)
        self.observation_space = spaces.Discrete(state_dim)

        db_max_access = None if not testing else queue_size + 2

        self._sampler = DataSampler(
            db_path=db_path,
            queue_size=queue_size,
            max_access=db_max_access,
            feature_params=feature_params,
            max_training_data=max_training_data,
        )

        self._policy_net = PolicyNet(
            input_dim=state_dim,
            output_dim=action_dim,
            position_dim=len(Position),
            embedding_dim=embedding_dim,
            num_mem=num_mem,
            mem_dim=mem_dim,
        ).to(device)

        self._manager = TradeManager(0, alpha, gamma, action_cost, leverage)

        self._leverage = leverage
        self._device = device
        self._sharpe_cutoff = sharpe_cutoff
        self._return_thresh = return_thresh
        self._inaction_cost = inaction_cost
        self._action_cost = action_cost
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._zeta = zeta
        self._portfolio = 1.0
        self._returns = []
        self._log_probs = []
        self._init_close = 0.0
        self._risk_free_rate = 1.0
        self._entry = 0.0
        self._exit = 0.0
        self._log_return = 0.0

    @property
    def beta(self):
        return self._beta

    @property
    def sampler(self):
        return self._sampler

    @property
    def model(self):
        """
        Gets the policy network model.

        Returns:
            torch.nn.Module: The policy network model used by the agent.
        """
        return self._policy_net

    @property
    def model_weights(self) -> Dict[str, Any]:
        """
        Retrieve the model weights from the policy network.

        Returns a dictionary containing the state of the policy network's parameters, mapping each parameter's name to its value. The policy network is set to evaluation mode before retrieval.

        Returns:
            Dict[str, Any]: A dictionary mapping parameter names to their values.
        """
        self._policy_net.eval()
        return self._policy_net.state_dict()

    @model_weights.setter
    def model_weights(self, state_dict: Dict[str, Any]) -> None:
        """
        Load weights into the policy network model from a state dictionary.

        This method sets the policy network to evaluation mode, moves it to CPU, loads the given
        state dictionary into the model, and then moves the model back to the original device.

        Parameters
        ----------
        state_dict : Dict[str, Any]
            The state dictionary containing the model weights to be loaded.

        Returns
        -------
        None
        """
        self._policy_net.eval()
        self._policy_net.to("cpu")
        self._policy_net.load_state_dict(state_dict)
        self._policy_net.to(self._device)

    @property
    def log_return(self) -> float:
        """
        Returns the logarithm of the return (log return) for the investment.

        The log return is a measure of performance that represents the logarithm of the ratio of the final value to the initial value.

        Returns:
            float: The logarithmic return value.
        """
        return self._log_return

    @property
    def log_probs(self) -> list:
        """
        Returns the log probabilities of actions taken during the episode.

        Returns:
            list: A list containing the log probabilities of actions taken in the current episode.
        """
        return self._log_probs

    @property
    def portfolio(self) -> float:
        """
        Get the current portfolio value.

        Returns:
            float: The current value of the portfolio.
        """
        return self._portfolio

    @property
    def init_close(self) -> float:
        """
        Get the initial close price when the environment was instantiated.

        Returns:
            float: The initial close price of the asset as a float value.
        """
        return self._init_close

    def reset(self):
        """
        Reset the environment to its initial state.
        This method initializes a new trading episode by:
        1. Resetting the sampler
        2. Sampling until a valid state is obtained
        3. Initializing a new TradeManager with the specified parameters
        4. Resetting all trading variables and positions
        Returns:
            list: The initial state representation for the new episode
        """
        self._sampler.reset()
        _, close, state = self._sampler.sample_next()

        while len(state) == 0:
            _, close, state = self._sampler.sample_next()

        self._manager = TradeManager(
            cov=self.sampler.coef_of_var,
            alpha=self._alpha,
            gamma=self._gamma,
            cost=self._action_cost,
            leverage=self._leverage,
        )

        self._position = Position.Cash
        self._portfolio = 1.0
        self._returns = []
        self._log_probs = []
        self._init_close = close
        self._risk_free_rate = 1.0
        self._entry = 0.0
        self._exit = 0.0
        self._log_return = 0.0
        return state

    def step(self, action: int):
        """
        Execute a step in the environment based on the provided action.
        The step function processes an action (Buy, Sell, or Hold), calculates rewards,
        updates environment state, and determines if the episode is done.
        Parameters:
            action (int): The action to take in the environment (0: Buy, 1: Sell, 2: Hold)
        Returns:
            tuple: Contains the following elements:
                - action (int): The next action selected by the policy network
                - reward (float): The reward obtained from the current action
                - done (bool): Whether the episode has ended
                - truncated (bool): Always False in this implementation
                - info (dict): Additional information including current price
        Reward Calculation:
            - Buy: Incurs action cost, with adjustments for doubling or forced selling
            - Sell: Calculates gain-based reward incorporating Sharpe ratio and trade size
            - Hold: Penalizes inaction based on potential gains or market movement
        """
        end, price, state = self._sampler.sample_next()
        reward, done = 0, False

        self._risk_free_rate = price / self._init_close
        # Validate buy
        if Action(action) == Action.Buy:
            a = self._manager.try_buy(price, self.sampler.coef_of_var)
            if a == Action.Double:
                reward -= self._action_cost * (self._manager.potential_gain(price) - 1)
            elif a == Action.Sell:
                done = True
                reward -= self._manager.curr_trade.leverage * (1 + self._action_cost)
            else:
                reward -= self._action_cost

        elif Action(action) == Action.Sell:
            gain = self._manager.try_sell(price, self.sampler.coef_of_var)
            if gain > 0:
                if self._manager.portfolio < self._return_thresh:
                    self._logger.info("portfolio threshold hit, episode done")
                    done = True

                self._log_return += np.log(gain)
                sharpe = compute_sharpe_ratio(
                    returns=self._manager.returns[-self._sharpe_cutoff :],
                    risk_free_rate=self._risk_free_rate * (1 - self._action_cost),
                )

                double = self._manager.last_trade["amount"] > 0.5
                reward += (1 - self._zeta) * sharpe + self._zeta * (gain - 1) * (
                    2 if double else 1
                )

        elif Action(action) == Action.Hold:
            if self._manager.asset or self._manager.partial:
                potential_gain = self._manager.potential_gain(price)
                reward -= self._inaction_cost * potential_gain
            else:
                if self._manager.cash:
                    prev_exit = self._manager.prev_exit
                    prev_exit = prev_exit if prev_exit > 0 else price
                    reward -= self._inaction_cost * (price / prev_exit)
                else:
                    reward -= (
                        self._inaction_cost
                        * self._risk_free_rate
                        * (1 - self._action_cost)
                    )

        action, log_prob = select_action(
            model=self._policy_net,
            state=state,
            potential=self._manager.potential_gain(price) - 1,
            position=int(self._position.value),
            device=self._device,
        )

        self._log_probs += [log_prob]

        if end:
            done = True

        return action, reward, done, False, {"price": price}


def train(
    env: TradeEnv,
    episodes: int,
    learning_rate: float = 1e-3,
    momentum: float = 0.9,
    weight_decay: float = 0.9,
    max_grad_norm: float = 1.0,
    min_episodes: int = 10,
) -> float:
    """
    Train the trading agent using reinforcement learning.
    This function performs the training loop for the trading agent, updating the model
    parameters based on the rewards received from the environment. It uses the policy
    gradient method to optimize the model's performance.
    Parameters:
        env (TradeEnv): The trading environment instance.
        episodes (int): Number of training episodes to run.
        learning_rate (float): Learning rate for the optimizer. Default is 1e-3.
        momentum (float): Momentum factor for the optimizer. Default is 0.9.
        weight_decay (float): Weight decay for the optimizer. Default is 0.9.
        max_grad_norm (float): Maximum gradient norm for clipping. Default is 1.0.
        min_episodes (int): Minimum number of episodes before early stopping. Default is 10.
    Returns:
        float: The ratio of the best portfolio value to the buy-and-hold strategy.
    """
    env._logger.info("training starts")
    optimizer = optim.SGD(
        env.model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    buy_and_hold = None
    best_model = {
        "result": -1,
        "weights": None,
    }

    for e in range(episodes):
        env._logger.info(f"episode {e+1}/{episodes} began")
        _ = env.reset()
        env.model.train()
        action, reward, done, _, close = env.step(1)
        rewards = [reward]
        while not done:
            action, reward, done, _, close = env.step(action)
            rewards += [reward]

        optimizer.zero_grad()
        log_return = env.log_return
        loss = compute_loss(
            log_probs=env.log_probs,
            rewards=rewards,
            beta=env.beta,
            log_return=log_return,
            device=env._device,
        )

        loss.backward()
        nn.utils.clip_grad_norm_(env.model.parameters(), max_grad_norm)
        optimizer.step()

        if not buy_and_hold:
            buy_and_hold = close["price"] / env.init_close

        env._logger.info(
            f"""\n
        episode {e+1}/{episodes} done
        loss:            {' ' if loss.item() > 0 else ''}{loss.item():.4f}
        model portfolio: {'+' if env._manager.portfolio > 1 else ''}{(env._manager.portfolio-1) * 100:.4f}%
        buy and hold:    {'+' if buy_and_hold > 1 else ''}{(buy_and_hold-1) * 100:.4f}%
        """
        )
        env.model.eval()

        if env._manager.portfolio > best_model["result"]:
            best_model["result"] = env._manager.portfolio
            best_model["weights"] = env.model_weights

        if env._manager.portfolio > max(1.0, buy_and_hold) and e + 1 >= min_episodes:
            env._logger.info(
                f"target reached, best episode achieved {'+' if best_model['result'] > 1 else ''}{(best_model['result']-1) * 100:.4f}%, exiting."
            )
            env.model_weights = best_model["weights"]
            return best_model["result"] / buy_and_hold
        else:
            if best_model["weights"]:
                env.model_weights = best_model["weights"]

    env._logger.info(
        f"training complete, best episode achieved {'+' if best_model['result'] > 1 else ''}{(best_model['result']-1) * 100:.4f}%"
    )
    env.model_weights = best_model["weights"]
    return best_model["result"] / buy_and_hold
