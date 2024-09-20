# main.py

import numpy as np
import time

from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch
from mctspy.games.tictactoe2 import TicTacToeGameState


def print_board(board):
    symbols = {0: " ", 1: "X", -1: "O"}
    print("\n" + "----" * board.shape[1] + "-")
    for row in board:
        print("|", end="")
        for cell in row:
            print(f" {symbols[cell]} |", end="")
        print("\n" + "----" * board.shape[1] + "-")


def play_game(board_size: int = 3, connect: int = 3, simulations_per_move: int = 10000):
    # Initialize the game state
    initial_board_state = np.zeros((board_size, board_size), dtype=int)
    state = TicTacToeGameState(state=initial_board_state, next_to_move=0, win=connect)

    game_start_time = time.time()
    move_count = 0

    while not state.is_game_over():
        move_start_time = time.time()

        # Initialize the root node for the current state
        root = TwoPlayersGameMonteCarloTreeSearchNode(state=state)

        # Initialize MCTS with the root node and specify the number of processes
        mcts = MonteCarloTreeSearch(root, num_processes=4)  # Adjust num_processes as needed

        # Perform parallel MCTS to find the best move
        best_node = mcts.best_action_parallel(simulations_number=simulations_per_move)

        if best_node is None:
            print("No valid moves available. Game ending.")
            break

        # Update the game state with the selected move
        state = best_node.state
        move_count += 1

        move_end_time = time.time()
        move_duration = move_end_time - move_start_time

        # Display the move information and current board state
        print(f"\nMove {move_count} (Player {'X' if state.next_to_move == 1 else 'O'}):")
        print(
            f"Action taken: "
            f"x:{best_node.parent_action.x_coordinate+1} y:{best_node.parent_action.y_coordinate+1}"
        )
        print(f"Simulations: {simulations_per_move}")
        print(f"Time taken: {move_duration:.2f} seconds")
        print_board(state.board)

    # After the game loop, print the game result
    game_end_time = time.time()
    game_duration = game_end_time - game_start_time

    print("\nGame Over!")
    print(f"Total moves: {move_count}")
    print(f"Total game time: {game_duration:.2f} seconds")

    result = state.game_result
    if result == 1:
        winner = "X"
    elif result == -1:
        winner = "O"
    else:
        winner = "Draw"
    print(f"Winner: {winner}")


if __name__ == "__main__":
    play_game(board_size=5, connect=4, simulations_per_move=20000)
