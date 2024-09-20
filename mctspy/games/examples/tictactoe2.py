# tictactoe2.py

from enum import Enum
import numpy as np

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
    def game_result(self):
        # check if game is over
        for i in range(self.board_size - self.win + 1):
            rowsum = np.sum(self.board[i : i + self.win], 0)
            colsum = np.sum(self.board[:, i : i + self.win], 1)
            if self.win in rowsum or self.win in colsum:
                return 0  # Player 0 wins
            if -self.win in rowsum or -self.win in colsum:
                return 1  # Player 1 wins
        for i in range(self.board_size - self.win + 1):
            for j in range(self.board_size - self.win + 1):
                sub = self.board[i : i + self.win, j : j + self.win]
                diag_sum_tl = sub.trace()
                diag_sum_tr = sub[::-1].trace()
                if diag_sum_tl == self.win or diag_sum_tr == self.win:
                    return 0  # Player 0 wins
                if diag_sum_tl == -self.win or diag_sum_tr == -self.win:
                    return 1  # Player 1 wins

        # draw
        if np.all(self.board != 0):
            return -1  # Draw

        # if not over - no result
        return None

    def is_game_over(self):
        return self.game_result is not None

    def is_move_legal(self, move: TicTacToeMove):
        # check if correct player moves
        if move.player != self.next_to_move:
            return False

        # check if inside the board
        if not (0 <= move.x_coordinate < self.board_size and 0 <= move.y_coordinate < self.board_size):
            return False

        # check if board field not occupied yet
        return self.board[move.x_coordinate, move.y_coordinate] == 0

    def move(self, move: TicTacToeMove):
        if not self.is_move_legal(move):
            raise ValueError(f"move {move} on board {self.board} is not legal")
        new_board = np.copy(self.board)
        new_board[move.x_coordinate, move.y_coordinate] = 1 if move.player == 0 else -1
        next_to_move = 1 - self.next_to_move  # Switch players
        return TicTacToeGameState(new_board, next_to_move, self.win)

    def get_legal_actions(self):
        indices = np.where(self.board == 0)
        return [
            TicTacToeMove(coords[0], coords[1], self.next_to_move) for coords in list(zip(indices[0], indices[1]))
        ]

    def get_player_relation(self, player1: int, player2: int) -> PlayerRelation:
        return PlayerRelation.ADVERSARIAL if player1 != player2 else PlayerRelation.COOPERATIVE
