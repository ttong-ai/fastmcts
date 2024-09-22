# FastMCTS: Fast Python Implementation of Monte Carlo Tree Search

FastMCTS is a high-performance Python library that implements the Monte Carlo Tree Search (MCTS) algorithm with parallel capabilities. Designed for efficiency, it's suitable for both small and large game trees. This implementation is inspired by the [Monte Carlo Tree Search Beginner's Guide](https://int8.io/monte-carlo-tree-search-beginners-guide) and enhanced for speed.

## Features

- Fast, parallel implementation of MCTS
- Easy-to-use interface
- Support for two-player zero-sum games
- Extensible for custom game implementations
- Includes examples for Tic-Tac-Toe and Connect Four
- Optimized for performance with parallel processing

## Installation

Install FastMCTS using pip:

```bash
pip install fastmcts
```

## Quick Start

Here's a simple example of how to use FastMCTS with Tic-Tac-Toe:

```python
import numpy as np
from fastmcts.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from fastmcts.tree.search import MonteCarloTreeSearch
from fastmcts.games.tictactoe import TicTacToeGameState

# Initialize game state
state = np.zeros((3, 3))
initial_board_state = TicTacToeGameState(state=state, next_to_move=1)

# Set up MCTS
root = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_board_state)
mcts = MonteCarloTreeSearch(root)

# Find the best action (utilizing parallel processing)
best_node = mcts.best_action_parallel(total_simulation_seconds=1)
```

## Using FastMCTS for Your Own Games

To use FastMCTS for your own two-player zero-sum game:

1. Create a new game state class that inherits from `fastmcts.games.common.TwoPlayersGameState`.
2. Implement the required methods (see `fastmcts.games.tictactoe.TicTacToeGameState` for an example).
3. Use your custom game state with the MCTS algorithm as shown in the quick start example.

## Example: Connect Four Game Play

Here's an example of how to use FastMCTS to play Connect Four:

```python
import numpy as np
from fastmcts.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from fastmcts.tree.search import MonteCarloTreeSearch
from fastmcts.games.connect4 import Connect4GameState

# Initialize game state
state = np.zeros((7, 7))
board_state = Connect4GameState(
    state=state, next_to_move=np.random.choice([-1, 1]), win=4)

# Define piece representations
pieces = {0: " ", 1: "X", -1: "O"}

# Helper functions to display the board
def stringify(row):
    return " " + " | ".join(map(lambda x: pieces[int(x)], row)) + " "

def display(board):
    board = board.copy().T[::-1]
    for row in board[:-1]:
        print(stringify(row))
        print("-" * (len(row) * 4 - 1))
    print(stringify(board[-1]))
    print()

# Main game loop
display(board_state.board)
while board_state.game_result is None:
    # Calculate best move using parallel processing
    root = TwoPlayersGameMonteCarloTreeSearchNode(state=board_state)
    mcts = MonteCarloTreeSearch(root)
    best_node = mcts.best_action_parallel(total_simulation_seconds=1)

    # Update and display board
    board_state = best_node.state
    display(board_state.board)

# Print result
print(f"Winner: {pieces[board_state.game_result]}")
```

## Performance

FastMCTS leverages parallel processing to significantly speed up the Monte Carlo Tree Search. This makes it suitable for more complex games and larger search spaces compared to traditional sequential MCTS implementations.

## Contributing

Contributions to FastMCTS are welcome! Please feel free to submit pull requests, create issues or suggest improvements. We're particularly interested in optimizations and extensions that can further improve performance.

## License

This project is licensed under the MIT License. See the LICENSE file for details.