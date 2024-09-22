# nodes.py

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Optional, Dict, Any
import numpy as np
import random

from fastmcts.games.common import (
    AbstractGameState,
    AbstractGameAction,
    TwoPlayersAbstractGameState,
    PlayerRelation,
    GeneralPlayerAbstractGameState,
    GeneralPlayerAbstractGameAction,
)
from fastmcts.utils import _pickle_method, _unpickle_method


class MonteCarloTreeSearchNode(ABC):
    """
    Abstract base class for a node in the Monte Carlo Tree Search (MCTS) algorithm.
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

    def best_child(self, c_param: float = 1.4) -> Optional["MonteCarloTreeSearchNode"]:
        if not self.children:
            return None

        choices_weights = []
        for child in self.children:
            if child.n == 0:
                weight = float("inf")
            else:
                weight = (child.q / child.n) + c_param * np.sqrt((2 * np.log(self.n)) / child.n)
            choices_weights.append(weight)
        best_child = self.children[np.argmax(choices_weights)]
        return best_child

    @staticmethod
    def rollout_policy(possible_moves: List[AbstractGameAction]) -> AbstractGameAction:
        return random.choice(possible_moves)


class TwoPlayersGameMonteCarloTreeSearchNode(MonteCarloTreeSearchNode):
    """
    A node in the Monte Carlo Tree Search for two-player games.
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
        if self.parent is None:
            # Root node: aggregate all results
            return sum(self._results.values())
        current_player = self.parent.state.next_to_move
        wins = self._results.get(1, 0) if current_player == 0 else self._results.get(-1, 0)
        loses = self._results.get(-1, 0) if current_player == 0 else self._results.get(1, 0)
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
    A node in the Monte Carlo Tree Search for general problem-solving.
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
        wins = self._results.get(1, 0) if current_player == 0 else self._results.get(-1, 0)
        loses = self._results.get(-1, 0) if current_player == 0 else self._results.get(1, 0)
        return wins - loses

    def _calculate_cooperative_q(self) -> float:
        return sum(self._results.values())

    def _calculate_mixed_q(self, current_player: int) -> float:
        player_score = self._results.get(1, 0) if current_player == 0 else self._results.get(-1, 0)
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
