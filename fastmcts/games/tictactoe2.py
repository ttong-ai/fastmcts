# tictactoe2.py
from typing import List, Optional
import numpy as np

from fastmcts.games.common import (
    PlayerRelation,
    GeneralPlayerAbstractGameAction,
    GeneralPlayerAbstractGameState,
)


class TicTacToeMove(GeneralPlayerAbstractGameAction):
    def __init__(self, x_coordinate: int, y_coordinate: int, player: int):
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.player = player

    def __repr__(self):
        return f"x:{self.x_coordinate} y:{self.y_coordinate} player:{self.player}"


class TicTacToeGameState(GeneralPlayerAbstractGameState):
    def __init__(self, state: np.ndarray, next_to_move: int = 0, win: int = None):
        if len(state.shape) != 2:
            raise ValueError("Only 2D boards allowed")
        self.board = state
        self.board_rows = state.shape[0]
        self.board_cols = state.shape[1]
        if win is None:
            win = min(self.board_rows, self.board_cols)
        self.win = win
        super().__init__(num_players=2, player_relations=PlayerRelation.ADVERSARIAL)
        self.next_to_move = next_to_move
        self.winning_sequence = None

    @property
    def game_result(self):
        result, sequence = self._check_winner()
        self.winning_sequence = sequence  # Set the winning sequence
        if result is not None:
            return result
        elif np.all(self.board != 0):
            # Draw
            return 0
        else:
            # Game is not over
            return None

    def _check_winner(self):
        # Check rows
        for row in range(self.board_rows):
            for col in range(self.board_cols - self.win + 1):
                window = self.board[row, col : col + self.win]
                if np.all(window == 1):
                    sequence = [(row, c) for c in range(col, col + self.win)]
                    return 1, sequence
                elif np.all(window == -1):
                    sequence = [(row, c) for c in range(col, col + self.win)]
                    return -1, sequence

        # Check columns
        for col in range(self.board_cols):
            for row in range(self.board_rows - self.win + 1):
                window = self.board[row : row + self.win, col]
                if np.all(window == 1):
                    sequence = [(r, col) for r in range(row, row + self.win)]
                    return 1, sequence
                elif np.all(window == -1):
                    sequence = [(r, col) for r in range(row, row + self.win)]
                    return -1, sequence

        # Check positive diagonals
        for row in range(self.board_rows - self.win + 1):
            for col in range(self.board_cols - self.win + 1):
                window = [self.board[row + i, col + i] for i in range(self.win)]
                if np.all(np.array(window) == 1):
                    sequence = [(row + i, col + i) for i in range(self.win)]
                    return 1, sequence
                elif np.all(np.array(window) == -1):
                    sequence = [(row + i, col + i) for i in range(self.win)]
                    return -1, sequence

        # Check negative diagonals
        for row in range(self.win - 1, self.board_rows):
            for col in range(self.board_cols - self.win + 1):
                window = [self.board[row - i, col + i] for i in range(self.win)]
                if np.all(np.array(window) == 1):
                    sequence = [(row - i, col + i) for i in range(self.win)]
                    return 1, sequence
                elif np.all(np.array(window) == -1):
                    sequence = [(row - i, col + i) for i in range(self.win)]
                    return -1, sequence

        # No winner
        return None, None

    def is_game_over(self) -> bool:
        return self.game_result is not None

    def is_move_legal(self, move: TicTacToeMove) -> bool:
        # Check if correct player moves
        if move.player != self.next_to_move:
            return False

        # Check if inside the board
        if not (0 <= move.x_coordinate < self.board_rows and 0 <= move.y_coordinate < self.board_cols):
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
