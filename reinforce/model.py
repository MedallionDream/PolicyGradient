import math
import torch
import logging
import torch.nn as nn
import numpy as np

from typing import Tuple, List


class MemoryNet(nn.Module):
    """
    A neural network module for memory-based attention mechanism.
    This class implements a memory network that uses attention to retrieve information from a fixed set of memory slots.
    The memory slots contain learnable parameters that are transformed into keys and values.
    Input queries are matched against the memory keys using attention to produce a weighted sum of memory values.
    Parameters
    ----------
    num_mem : int
        Number of memory slots in the network.
    mem_dim : int
        Dimension of each memory slot.
    inp_dim : int
        Dimension of the input query vectors and output vectors.
    Methods
    -------
    forward(k: torch.Tensor) -> torch.Tensor
        Process input queries through the memory network.
        Parameters:
        k : torch.Tensor
            Input query tensor of shape [n x d], where n is the batch size and d is inp_dim.
        Returns:
        torch.Tensor
            Memory-augmented output tensor of shape [n x d].
    """

    def __init__(self, num_mem: int, mem_dim: int, inp_dim: int) -> None:
        super().__init__()
        self._mem = nn.Parameter(torch.randn((num_mem, mem_dim), dtype=torch.float32))
        self._fk = nn.Linear(mem_dim, inp_dim)
        self._fv = nn.Linear(mem_dim, inp_dim)

        # Initialize parameters
        nn.init.kaiming_uniform_(self._fk.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self._fv.weight, a=math.sqrt(5))

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the attention mechanism.

        This method computes attention scores between the input tensor k and the
        internal memory, applies softmax to get attention weights, and returns
        a weighted sum of value vectors derived from the memory.

        Args:
            k (torch.Tensor): Input tensor of shape [n x d], where n is batch size
                            and d is the feature dimension.

        Returns:
            torch.Tensor: Output tensor of shape [n x d] representing the attended values.

        Process:
            1. Transforms memory into keys and values
            2. Computes attention scores between input and keys
            3. Normalizes scores using softmax
            4. Produces output as weighted sum of values
        """
        key = torch.softmax(self._fk(self._mem), dim=-1)  # [h x d]
        val = torch.relu(self._fv(self._mem))  # [h x d]

        # Compute attention scores
        att = torch.matmul(k, key.t())  # [n x h]

        # Apply softmax to get attention weights
        att_weights = torch.softmax(att, dim=-1)  # [n x h]

        # Compute the weighted sum of the values
        output = torch.matmul(att_weights, val)  # [n x d]

        return output


class PolicyNet(nn.Module):
    """
    A neural network module for policy-based reinforcement learning.

    This network combines embedding, feed-forward layers, and memory retrieval 
    to process input features and position information, ultimately outputting
    action probabilities.

        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output (number of possible actions).
        position_dim (int): Vocabulary size for position embeddings.
        embedding_dim (int): Dimension of position embeddings.
        num_mem (int, optional): Number of memory slots in MemoryNet. Defaults to 512.
        mem_dim (int, optional): Dimension of each memory slot. Defaults to 256.

    Architecture:
        1. Embeds position indices using an embedding layer
        2. Concatenates input features, position embeddings, and portfolio features
        3. Passes through two fully connected layers with ReLU activation, 
            layer normalization, and dropout
        4. Generates key and value vectors for memory retrieval
        5. Queries the memory network and concatenates the result with value vectors
        6. Outputs softmax probabilities over possible actions
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        position_dim: int,
        embedding_dim: int,
        num_mem: int = 512,
        mem_dim: int = 256,
    ) -> None:
        super().__init__()

        self._embedding = nn.Embedding(position_dim, embedding_dim)
        self._f1 = nn.Linear(input_dim + embedding_dim + 1, 512)
        self._f2 = nn.Linear(512, 256)
        self._fk = nn.Linear(256, 128)
        self._fv = nn.Linear(256, 128)
        self._f4 = nn.Linear(256, output_dim)
        self._mem = MemoryNet(num_mem, mem_dim, 128)

        # Adding dropout and normalization layers
        self.dropout = nn.Dropout(p=0.5)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(256)

    def forward(
        self, x: torch.Tensor, p: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the PolicyNet.

        Parameters:
        x (torch.Tensor): Input features tensor.
        p (torch.Tensor): Position indices tensor.
        g (torch.Tensor): Portfolio features tensor.

        Returns:
        torch.Tensor: Output tensor after passing through the network.
        """
        x = torch.cat((x, self._embedding(p).squeeze(1), g), dim=-1)
        x = self.norm1(torch.relu(self._f1(x)))
        x = self.dropout(x)
        x = self.norm2(torch.relu(self._f2(x)))
        x = self.dropout(x)

        k = torch.softmax(self._fk(x), dim=-1)
        v = torch.relu(self._fv(x))
        q = self._mem(k)
        x = torch.cat((v, q), dim=-1)

        return torch.softmax(self._f4(x), dim=-1)


def select_action(
    model: nn.Module, state: np.ndarray, potential: float, position: int, device: str
) -> Tuple[int, torch.Tensor]:
    """
    Selects an action based on the model's output probabilities.

    This function passes the state, position, and potential through the model to get action probabilities.
    Then it randomly samples an action according to these probabilities.

    Args:
        model (nn.Module): The neural network model that produces action probabilities.
        state (np.ndarray): The current state or observation from the environment.
        potential (float): The potential value associated with the current state.
        position (int): The position identifier or index.
        device (str): The device to run the model on ('cpu' or 'cuda').

    Returns:
        Tuple[int, torch.Tensor]: A tuple containing:
            - The selected action as an integer
            - The log probability of the selected action
    """
    probs = model(
        x=torch.tensor(state, dtype=torch.float32, device=device),
        p=torch.tensor([[position]], dtype=torch.long, device=device),
        g=torch.tensor([[potential]], dtype=torch.float32, device=device),
    )

    action = np.random.choice(probs.size(-1), p=probs.detach().cpu().numpy()[0])
    return action, torch.log(probs[0, action])


def compute_discounted_rewards(rewards: List[float], gamma: float) -> List[float]:
    """
    Compute discounted rewards for a sequence of rewards.

    This function calculates the discounted cumulative rewards for each time step in a sequence,
    applying the discount factor gamma to future rewards.

    Args:
        rewards (List[float]): A list of reward values received at each time step.
        gamma (float): The discount factor (between 0 and 1) that determines how much 
                    future rewards are valued compared to immediate rewards.

    Returns:
        List[float]: A list of discounted cumulative rewards, where each value represents
                    the sum of the current reward and all future rewards, with future rewards
                    discounted by gamma raised to the power of their distance in the future.

    Example:
        >>> compute_discounted_rewards([1, 0, 1], 0.9)
        [1.81, 0.9, 1.0]
    """

    discounted_rewards = []
    cumulative = 0
    for reward in reversed(rewards):
        cumulative = reward + gamma * cumulative
        discounted_rewards.insert(0, cumulative)
    return discounted_rewards


def compute_loss(
    log_probs: List[torch.Tensor],
    rewards: List[float],
    gamma: float = 0.99,
    log_return: float | None = None,
    beta: float | None = 0.5,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute the loss for policy gradient methods (REINFORCE).

    This function calculates the policy gradient loss using the collected log probabilities
    and rewards from an episode. It applies discount factors to rewards and normalizes them
    for stable training.

    Args:
        log_probs (List[torch.Tensor]): List of log probabilities of actions taken during the episode.
        rewards (List[float]): List of rewards received at each step in the episode.
        gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
        log_return (float | None, optional): Logarithm of the return for baseline adjustment. 
                                            Defaults to None.
        beta (float | None, optional): Weighting factor for balancing between normalized rewards
                                    and log return when using a baseline. Defaults to 0.5.
        device (str, optional): Device to perform tensor operations on. Defaults to "cpu".

    Returns:
        torch.Tensor: Computed policy gradient loss as a scalar tensor.

    Note:
        When log_return and beta are provided, the function implements a weighted mixture
        between standard REINFORCE and a baseline method, which can reduce variance.
    """
    log_probs = torch.stack(log_probs)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

    discounted_rewards = compute_discounted_rewards(rewards.tolist(), gamma)
    discounted_rewards = torch.tensor(
        discounted_rewards, dtype=torch.float32, device=device
    )
    normalized_rewards = (discounted_rewards - discounted_rewards.mean()) / (
        discounted_rewards.std() + 1e-9
    )

    if log_return is not None and beta is not None:
        policy_gradient = -log_probs * (
            beta * normalized_rewards + (1 - beta) * log_return
        )
    else:
        policy_gradient = -log_probs * normalized_rewards

    return policy_gradient.sum()


def inference(
    model: nn.Module,
    state: np.ndarray,
    position: int,
    potential: float,
    device: str = "cpu",
    method: str = "argmax",
    min_prob: float = 0.33,
) -> Tuple[int, float]:
    """
    Performs inference using a trained neural network model.

    This function takes a trained model and relevant input data to predict an action. It supports
    two inference methods: 'argmax' (deterministic, selecting the highest probability action) or
    'prob' (probabilistic sampling based on output distribution).

    Args:
        model (nn.Module): The trained neural network model.
        state (np.ndarray): The current state representation as input to the model.
        position (int): The position value to be passed to the model.
        potential (float): The potential value to be passed to the model.
        device (str, optional): Device to run inference on ('cpu' or 'cuda'). Defaults to "cpu".
        method (str, optional): Inference method, either 'argmax' or 'prob'. Defaults to "argmax".
        min_prob (float, optional): Minimum probability threshold for the 'prob' method. If the 
                                selected action has a probability below this threshold, 
                                defaults to argmax selection. Defaults to 0.31.

    Returns:
        Tuple[int, float]: A tuple containing:
            - The selected action (int)
            - The probability of the selected action (float)

    Raises:
        ValueError: If an invalid method is provided.
    """
    model.eval()
    min_prob = min(0.34, min_prob)

    if method not in {"argmax", "prob"}:
        logging.error("Method must be in one of [argmax, prob]")
        return 1

    with torch.no_grad():
        probs = model(
            x=torch.tensor(state, dtype=torch.float32, device=device),
            p=torch.tensor([[position]], dtype=torch.long, device=device),
            g=torch.tensor([[potential]], dtype=torch.float32, device=device),
        )

    if method == "argmax":
        action = probs.argmax(1).item()
    else:
        action = np.random.choice(probs.size(-1), p=probs.detach().cpu().numpy()[0])

    prob = probs[0, action].item()
    if method == "prob" and prob < min_prob:
        action = probs.argmax(1).item()
        prob = probs[0, action].item()

    return action, prob
