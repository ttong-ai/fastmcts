# profile_mcts.py
import cProfile
import pstats
from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch
from mctspy.games.examples.tictactoe2 import TicTacToeGameState
import numpy as np


def run_mcts():
    initial_state = TicTacToeGameState(np.zeros((3, 3), dtype=int))
    root_node = TwoPlayersGameMonteCarloTreeSearchNode(initial_state)
    mcts = MonteCarloTreeSearch(root_node)
    best_node = mcts.best_action(simulations_number=1000)
    print(f"Best action leads to node with Q: {best_node.q}, N: {best_node.n}")


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    run_mcts()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(10)  # Print top 10 functions by cumulative time
