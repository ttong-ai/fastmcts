# tictactoe2.py

from enum import Enum
import numpy as np
from typing import List, Optional

from mctspy.games.common import PlayerRelation, GeneralPlayerAbstractGameAction, GeneralPlayerAbstractGameState


class TicTacToeMove(GeneralPlayerAbstractGameAction):
    def __init__(self, x_coordinate: int, y_coordinate: int, player: int):
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.player = player

    def __repr__(self):
        return f"x:{self.x_coordinate} y:{self.y_coordinate} player:{self.player}"


class TicTacToeGameState(GeneralPlayerAbstractGameState):
    def __init__(self, state: np.array, next_to_move: int = 0, win: int = None):
        if len(state.shape) != 2 or state.shape[0] != state.shape[1]:
            raise ValueError("Only 2D square boards allowed")
        self.board = state
        self.board_size = state.shape[0]
        if win is None:
            win = self.board_size
        self.win = win
        super().__init__(num_players=2, player_relations=PlayerRelation.ADVERSARIAL)
        self.next_to_move = next_to_move

    @property
    def game_result(self) -> Optional[int]:
        # Check for horizontal and vertical wins
        for i in range(self.board_size):
            row_sum = np.sum(self.board[i, :])
            if row_sum == self.win:
                return 1  # Player 0 wins
            elif row_sum == -self.win:
                return -1  # Player 1 wins

            col_sum = np.sum(self.board[:, i])
            if col_sum == self.win:
                return 1
            elif col_sum == -self.win:
                return -1

        # Check for diagonal wins
        diag_sum_tl = np.trace(self.board)
        if diag_sum_tl == self.win:
            return 1
        elif diag_sum_tl == -self.win:
            return -1

        diag_sum_tr = np.trace(np.fliplr(self.board))
        if diag_sum_tr == self.win:
            return 1
        elif diag_sum_tr == -self.win:
            return -1

        # Check for draw
        if not np.any(self.board == 0):
            return 0  # Draw

        # Game is not over
        return None

    def is_game_over(self) -> bool:
        return self.game_result is not None

    def is_move_legal(self, move: TicTacToeMove) -> bool:
        # Check if correct player moves
        if move.player != self.next_to_move:
            return False

        # Check if inside the board
        if not (0 <= move.x_coordinate < self.board_size and 0 <= move.y_coordinate < self.board_size):
            return False

        # Check if board field not occupied yet
        return self.board[move.x_coordinate, move.y_coordinate] == 0

    def move(self, move: TicTacToeMove) -> "TicTacToeGameState":
        if not self.is_move_legal(move):
            raise ValueError(f"Move {move} on board\n{self.board}\nis not legal")
        new_board = np.copy(self.board)
        new_board[move.x_coordinate, move.y_coordinate] = 1 if move.player == 0 else -1
        next_to_move = 1 - self.next_to_move  # Switch players
        return TicTacToeGameState(new_board, next_to_move, self.win)

    def get_legal_actions(self) -> List[TicTacToeMove]:
        indices = np.argwhere(self.board == 0)
        return [TicTacToeMove(x, y, self.next_to_move) for x, y in indices]

    def get_player_relation(self, player1: int, player2: int) -> PlayerRelation:
        return PlayerRelation.ADVERSARIAL if player1 != player2 else PlayerRelation.COOPERATIVE
