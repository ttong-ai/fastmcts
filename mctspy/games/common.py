from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Any, Dict, Union


class PlayerRelation(Enum):
    ADVERSARIAL = 1
    COOPERATIVE = 2
    MIXED = 3


class AbstractGameAction(ABC):
    pass


class GeneralPlayerAbstractGameAction(AbstractGameAction, ABC):
    pass


class AbstractGameState(ABC):

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


class GeneralPlayerAbstractGameState(AbstractGameState, ABC):

    def __init__(self, num_players: int, player_relations: Union[PlayerRelation, Dict[int, PlayerRelation]]):
        """
        Initialize the game state with player information.

        Parameters
        ----------
        num_players : int
            The number of players in the game.
        player_relations : Union[PlayerRelation, Dict[int, PlayerRelation]]
            The relationships between players. Can be a single PlayerRelation if all players
            have the same relationship, or a dictionary mapping player indices to their relations.
        """
        self.num_players = num_players
        self.next_to_move: int = 0  # Player index, 0 to num_players - 1

        if isinstance(player_relations, PlayerRelation):
            self.player_relations = {i: player_relations for i in range(num_players)}
        else:
            self.player_relations = player_relations

    def get_player_relation(self, player1: int, player2: int) -> PlayerRelation:
        """
        Get the relationship between two players.

        Parameters
        ----------
        player1 : int
            Index of the first player.
        player2 : int
            Index of the second player.

        Returns
        -------
        PlayerRelation
            The relationship between the two players.
        """
        if player1 == player2:
            return PlayerRelation.COOPERATIVE
        return self.player_relations[player1]

    def is_adversarial(self) -> bool:
        """
        Check if the game is purely adversarial (all players are adversaries).

        Returns
        -------
        bool
            True if all players are adversaries, False otherwise.
        """
        return all(relation == PlayerRelation.ADVERSARIAL for relation in self.player_relations.values())

    def is_cooperative(self) -> bool:
        """
        Check if the game is purely cooperative (all players are cooperating).

        Returns
        -------
        bool
            True if all players are cooperating, False otherwise.
        """
        return all(relation == PlayerRelation.COOPERATIVE for relation in self.player_relations.values())


class TwoPlayersAbstractGameState(AbstractGameState):

    @abstractmethod
    def __init__(self):
        self.next_to_move: int = 0  # Player index, 0 or 1

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
