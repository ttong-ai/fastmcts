from abc import ABC, abstractmethod
from typing import List, Optional, Any


class AbstractGameAction(ABC):
    pass


class AbstractGameState(ABC):

    next_to_move: int

    @abstractmethod
    def is_game_over(self) -> bool:
        """
        Indicates whether the game has ended.

        Returns
        -------
        bool
            True if the game is over, False otherwise.
        """
        pass

    @abstractmethod
    def game_result(self) -> Any:
        """
        Returns the result of the game.

        The specific return value depends on the game implementation.
        For example, it could be a score, a winner, or any other
        representation of the game's outcome.

        Returns
        -------
        Any
            The result of the game.
        """
        pass

    @abstractmethod
    def move(self, action: AbstractGameAction) -> "AbstractGameState":
        """
        Applies the given action to the current state and returns the resulting state.

        Parameters
        ----------
        action : AbstractGameAction
            The action to apply to the current state.

        Returns
        -------
        AbstractGameState
            The new game state after applying the action.
        """
        pass

    @abstractmethod
    def get_legal_actions(self) -> List[AbstractGameAction]:
        """
        Returns a list of legal actions from the current state.

        Returns
        -------
        List[AbstractGameAction]
            A list of legal actions that can be taken from the current state.
        """
        pass


class TwoPlayersAbstractGameState(AbstractGameState):

    @abstractmethod
    def game_result(self) -> Optional[int]:
        """
        Returns the result of the game for a two-player game.

        Returns
        -------
        Optional[int]
            1 if player #1 wins
            -1 if player #2 wins
            0 if there is a draw
            None if result is unknown
        """
        pass

    @abstractmethod
    def move(self, action: AbstractGameAction) -> "TwoPlayersAbstractGameState":
        """
        Applies the given action to the current state and returns the resulting state.

        Parameters
        ----------
        action : AbstractGameAction
            The action to apply to the current state.

        Returns
        -------
        TwoPlayersAbstractGameState
            The new game state after applying the action.
        """
        pass
