# profile_mcts.py
import cProfile
import numpy as np
import pstats

from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch
from mctspy.games.tictactoe2 import TicTacToeGameState


def run_mcts():
    initial_state = TicTacToeGameState(np.zeros((5, 5), dtype=int))
    root_node = TwoPlayersGameMonteCarloTreeSearchNode(initial_state)
    mcts = MonteCarloTreeSearch(root_node, num_processes=4)  # Specify number of processes
    best_node = mcts.best_action(simulations_number=10000)
    if best_node:
        print(f"Best action leads to node with Q: {best_node.q}, N: {best_node.n}")
    else:
        print("No valid moves found.")


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    run_mcts()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(10)  # Print top 10 functions by cumulative time
