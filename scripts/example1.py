import numpy as np
from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch
from mctspy.games.examples.tictactoe import TicTacToeGameState

# Create a 3x3 numpy array filled with zeros
initial_board_state = np.zeros((3, 3), dtype=int)

initial_state = TicTacToeGameState(state=initial_board_state, next_to_move=1)
root = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_state)
mcts = MonteCarloTreeSearch(root)
best_node = mcts.best_action(100)

print(f"Best action: {best_node.state.move}")
print(f"Resulting board state:\n{best_node.state.board}")
