# nodes.py
from abc import ABC, abstractmethod
from collections import defaultdict
from numba import jit
import numpy as np

from mctspy.games.common import AbstractGameState, TwoPlayersAbstractGameState


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
    children : list
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
        """
        Parameters
        ----------
        state : mctspy.games.common.AbstractGameState
        parent : MonteCarloTreeSearchNode
        """
        self.state: AbstractGameState = state
        self.parent: "MonteCarloTreeSearchNode" = parent
        self.children = []

    @property
    @abstractmethod
    def untried_actions(self):
        """

        Returns
        -------
        list of mctspy.games.common.AbstractGameAction

        """
        pass

    @property
    @abstractmethod
    def q(self):
        pass

    @property
    @abstractmethod
    def n(self):
        pass

    @abstractmethod
    def expand(self):
        pass

    @abstractmethod
    def is_terminal_node(self):
        pass

    @abstractmethod
    def rollout(self):
        pass

    @abstractmethod
    def backpropagate(self, reward):
        pass

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    @staticmethod
    @jit(nopython=True)
    def _uct_select(node_total_reward, node_visit_count, parent_visit_count, c_param):
        return (node_total_reward / node_visit_count) + c_param * np.sqrt(
            (2 * np.log(parent_visit_count) / node_visit_count)
        )

    def best_child(self, c_param=1.4):
        choices_weights = [self._uct_select(c.q, c.n, self.n, c_param) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    @staticmethod
    def rollout_policy(possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]


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
        self, state: TwoPlayersAbstractGameState, parent: MonteCarloTreeSearchNode = None, parent_action=None
    ):
        super().__init__(state, parent)
        self._number_of_visits = 0.0
        self._results = defaultdict(int)
        self._untried_actions = None
        self.parent_action = parent_action  # Add this line to store the action that led to this node

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    @property
    def q(self):
        wins = self._results[self.parent.state.next_to_move]
        loses = self._results[-1 * self.parent.state.next_to_move]
        return wins - loses

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        child_node = TwoPlayersGameMonteCarloTreeSearchNode(
            next_state, parent=self, parent_action=action  # Pass the action to the child node
        )
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result

    def backpropagate(self, result):
        self._number_of_visits += 1.0
        self._results[result] += 1.0
        if self.parent:
            self.parent.backpropagate(result)
