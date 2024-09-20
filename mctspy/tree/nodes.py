# nodes.py
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Optional, Dict, Any
import numpy as np
import random
import types

from mctspy.games.common import (
    AbstractGameState,
    AbstractGameAction,
    TwoPlayersAbstractGameState,
    PlayerRelation,
    GeneralPlayerAbstractGameState,
    GeneralPlayerAbstractGameAction,
)
from mctspy.utils import _pickle_method, _unpickle_method, copyreg


class MonteCarloTreeSearchNode(ABC):
    """
    Abstract base class for a node in the Monte Carlo Tree Search (MCTS) algorithm.

    This class defines the interface and basic structure for nodes used in MCTS.
    It provides abstract methods that must be implemented by concrete subclasses
    to define the specific behavior for different types of games or problems.

    Parameters
    ----------
    state : AbstractGameState
        The game state or problem state represented by this node.
    parent : MonteCarloTreeSearchNode, optional
        The parent node in the search tree. None for the root node.

    Attributes
    ----------
    state : AbstractGameState
        The game state or problem state represented by this node.
    parent : MonteCarloTreeSearchNode or None
        The parent node in the search tree.
    children : list of MonteCarloTreeSearchNode
        List of child nodes.

    Methods
    -------
    untried_actions (property)
        Returns a list of actions not yet explored from this node.
    q (property)
        Returns the total reward of the node.
    n (property)
        Returns the number of times the node has been visited.
    expand()
        Creates a new child node.
    is_terminal_node()
        Checks if the node represents a terminal state.
    rollout()
        Simulates a game or problem from this node to a terminal state.
    backpropagate(reward)
        Updates the node and its ancestors with a result.
    is_fully_expanded()
        Checks if all possible actions from this node have been explored.
    best_child(c_param=1.4)
        Selects the best child node based on the UCT formula.
    rollout_policy(possible_moves)
        Selects a move for the rollout phase of MCTS.

    Notes
    -----
    This abstract base class should be subclassed to implement MCTS for specific
    games or problems. The subclass must implement all abstract methods and can
    override other methods if needed.
    """

    def __init__(self, state: AbstractGameState, parent: "MonteCarloTreeSearchNode" = None):
        self.state: AbstractGameState = state
        self.parent: "MonteCarloTreeSearchNode" = parent
        self.children: List["MonteCarloTreeSearchNode"] = []

    @property
    @abstractmethod
    def untried_actions(self) -> List[AbstractGameAction]:
        pass

    @property
    @abstractmethod
    def q(self) -> float:
        pass

    @property
    @abstractmethod
    def n(self) -> float:
        pass

    @abstractmethod
    def expand(self) -> "MonteCarloTreeSearchNode":
        pass

    @abstractmethod
    def is_terminal_node(self) -> bool:
        pass

    @abstractmethod
    def rollout(self) -> Any:
        pass

    @abstractmethod
    def backpropagate(self, reward: Any) -> None:
        pass

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    @staticmethod
    def _uct_select(
        node_total_reward: float, node_visit_count: float, parent_visit_count: float, c_param: float
    ) -> float:
        return (node_total_reward / node_visit_count) + c_param * np.sqrt(
            (2 * np.log(parent_visit_count) / node_visit_count)
        )

    def best_child(self, c_param: float = 1.4) -> Optional["MonteCarloTreeSearchNode"]:
        if not self.children:
            return None

        choices_weights = []
        for child in self.children:
            if child.n == 0:
                weight = float("inf")
            else:
                weight = self._uct_select(child.q, child.n, self.n, c_param)
            choices_weights.append(weight)
        best_child = self.children[np.argmax(choices_weights)]
        return best_child

    @staticmethod
    def rollout_policy(possible_moves: List[AbstractGameAction]) -> AbstractGameAction:
        return random.choice(possible_moves)


class TwoPlayersGameMonteCarloTreeSearchNode(MonteCarloTreeSearchNode):
    """
    A node in the Monte Carlo Tree Search for two-player games.

    This class extends the MonteCarloTreeSearchNode to specifically handle
    two-player game scenarios. It implements the necessary methods for
    node expansion, simulation (rollout), and backpropagation of results.

    Parameters
    ----------
    state : mctspy.games.common.TwoPlayersAbstractGameState
        The game state represented by this node.
    parent : MonteCarloTreeSearchNode, optional
        The parent node in the search tree. None for the root node.
    parent_action : mctspy.games.common.AbstractGameAction, optional
        The action that led to this node from its parent. None for the root node.

    Attributes
    ----------
    _number_of_visits : float
        The number of times this node has been visited during the search.
    _results : collections.defaultdict
        A dictionary storing the results of simulations from this node.
    _untried_actions : list
        A list of legal actions that have not yet been tried from this state.

    Methods
    -------
    expand()
        Creates a new child node by taking an untried action.
    rollout()
        Performs a simulation from this node to a terminal state.
    backpropagate(result)
        Updates this node and its ancestors with the result of a simulation.
    """

    def __init__(
        self,
        state: TwoPlayersAbstractGameState,
        parent: Optional["TwoPlayersGameMonteCarloTreeSearchNode"] = None,
        parent_action: Optional[AbstractGameAction] = None,
    ):
        super().__init__(state, parent)
        self.state = state
        self.parent = parent
        self._number_of_visits: float = 0.0
        self._results: Dict[Any, float] = defaultdict(float)
        self._untried_actions: Optional[List[AbstractGameAction]] = None
        self.parent_action = parent_action

    @property
    def untried_actions(self) -> List[AbstractGameAction]:
        if self._untried_actions is None:
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    @property
    def q(self) -> float:
        current_player = self.parent.state.next_to_move if self.parent else self.state.next_to_move
        wins = self._results.get(current_player, 0)
        loses = self._results.get(-current_player, 0)
        return wins - loses

    @property
    def n(self) -> float:
        return self._number_of_visits

    def expand(self) -> "TwoPlayersGameMonteCarloTreeSearchNode":
        action = self.untried_actions.pop()
        next_state: TwoPlayersAbstractGameState = self.state.move(action)
        child_node = TwoPlayersGameMonteCarloTreeSearchNode(next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self) -> bool:
        return self.state.is_game_over()

    def rollout(self) -> Any:
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            if not possible_moves:
                break  # No possible moves
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result

    def backpropagate(self, reward: Any) -> None:
        self._number_of_visits += 1.0
        self._results[reward] += 1.0
        if self.parent:
            self.parent.backpropagate(reward)


class GeneralMonteCarloTreeSearchNode(MonteCarloTreeSearchNode):
    """
    A node in the Monte Carlo Tree Search for general problem-solving,
    including single-player puzzles, multi-player games, and optimization problems.

    This class extends the MonteCarloTreeSearchNode to handle a wide range of problems.
    It implements the necessary methods for node expansion, simulation (rollout),
    and backpropagation of results.

    The class is designed to work with various types of problem states and actions,
    including but not limited to:
    - Single-player puzzles and optimization problems (e.g., maze solving, scheduling)
    - Multi-player games (both cooperative and adversarial)
    - General state-space search problems

    It can be used for:
    - Finding optimal solutions in complex state spaces
    - Game playing (from simple board games to complex strategy games)
    - Decision-making under uncertainty
    - Optimization of processes or resource allocation

    The implementation works with any problem that can be modeled as a sequence of
    decisions or actions leading to a final state with a measurable outcome.

    Attributes:
    -----------
    state : GeneralPlayerAbstractGameState
        The problem state represented by this node.
    parent : GeneralMonteCarloTreeSearchNode, optional
        The parent node in the search tree. None for the root node.
    parent_action : GeneralPlayerAbstractGameAction, optional
        The action that led to this node from its parent. None for the root node.
    children : list of GeneralMonteCarloTreeSearchNode
        The child nodes of this node in the search tree.
    _number_of_visits : float
        The number of times this node has been visited during the search.
    _results : defaultdict
        A dictionary storing the results of simulations from this node.
    _untried_actions : list of GeneralPlayerAbstractGameAction
        A list of legal actions that have not yet been tried from this state.

    Methods:
    --------
    expand()
        Creates a new child node by taking an untried action.
    rollout()
        Performs a simulation from this node to a terminal state.
    backpropagate(result)
        Updates this node and its ancestors with the result of a simulation.
    is_terminal_node()
        Checks if the state represented by this node is a terminal state.
    is_fully_expanded()
        Checks if all possible actions from this state have been explored.
    best_child(c_param=1.4)
        Selects the best child node based on the UCT formula.
    rollout_policy(possible_moves)
        Selects a move for the rollout phase of MCTS.
    get_game_result(state)
        Retrieves the result from a given state.

    The class uses a flexible approach to handle different types of problems:
    - For single-player puzzles, it focuses on finding the optimal solution path.
    - For adversarial games, it considers the perspective of each player.
    - For cooperative games, it aims to maximize the overall outcome.
    - For optimization problems, it seeks to maximize or minimize an objective function.

    This implementation allows the MCTS algorithm to adapt its strategy based on
    the type of problem being solved, making it suitable for a wide range of
    applications in artificial intelligence, operations research, and decision theory.
    """

    def __init__(
        self,
        state: GeneralPlayerAbstractGameState,
        parent: Optional["GeneralMonteCarloTreeSearchNode"] = None,
        parent_action: Optional[GeneralPlayerAbstractGameAction] = None,
    ):
        super().__init__(state, parent)
        self.state = state
        self.parent = parent
        self._number_of_visits: float = 0.0
        self._results: Dict[Any, float] = defaultdict(float)
        self._untried_actions: Optional[List[GeneralPlayerAbstractGameAction]] = None
        self.parent_action = parent_action

    @property
    def untried_actions(self) -> List[AbstractGameAction]:
        if self._untried_actions is None:
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    @property
    def q(self) -> float:
        if self.parent is None:
            return sum(self._results.values())

        current_player = self.parent.state.next_to_move
        player_relation = self.parent.state.get_player_relation(current_player, self.state.next_to_move)

        if player_relation == PlayerRelation.ADVERSARIAL:
            return self._calculate_adversarial_q(current_player)
        elif player_relation == PlayerRelation.COOPERATIVE:
            return self._calculate_cooperative_q()
        else:  # MIXED
            return self._calculate_mixed_q(current_player)

    def _calculate_adversarial_q(self, current_player: int) -> float:
        wins = self._results.get(current_player, 0)
        loses = sum(self._results.get(p, 0) for p in range(self.state.num_players) if p != current_player)
        return wins - loses

    def _calculate_cooperative_q(self) -> float:
        return sum(self._results.values())

    def _calculate_mixed_q(self, current_player: int) -> float:
        player_score = self._results.get(current_player, 0)
        total_score = sum(self._results.values())
        return 2 * player_score - total_score  # Balances own score with overall outcome

    @property
    def n(self) -> float:
        return self._number_of_visits

    def expand(self) -> "GeneralMonteCarloTreeSearchNode":
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        child_node = GeneralMonteCarloTreeSearchNode(next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self) -> bool:
        return self.state.is_game_over()

    @staticmethod
    def get_game_result(state):
        return state.game_result() if callable(getattr(state, "game_result", None)) else state.game_result

    def rollout(self) -> Any:
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            if not possible_moves:
                break  # No possible moves
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return self.get_game_result(current_rollout_state)

    def backpropagate(self, result: Any) -> None:
        self._number_of_visits += 1.0
        self._results[result] += 1.0
        if self.parent:
            self.parent.backpropagate(result)
