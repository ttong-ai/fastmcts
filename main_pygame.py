# main_pygame.py

import pygame
import sys
import numpy as np
import time

from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch
from mctspy.games.tictactoe2 import TicTacToeGameState, TicTacToeMove

# Initialize Pygame
pygame.init()

# Constants
BOARD_SIZE = 5  # 5x5 board
CONNECT = 4  # Connect 4
CELL_SIZE = 100  # Size of each cell in pixels
MARGIN = 5  # Margin between cells
WINDOW_SIZE = (
    BOARD_SIZE * CELL_SIZE + (BOARD_SIZE + 1) * MARGIN,
    BOARD_SIZE * CELL_SIZE + (BOARD_SIZE + 1) * MARGIN + 100,
)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Fonts
FONT = pygame.font.SysFont(None, 40)
END_FONT = pygame.font.SysFont(None, 60)

# Initialize the display
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("MCTS Tic-Tac-Toe")

# Clock for controlling the frame rate
clock = pygame.time.Clock()


def draw_board(board):
    """
    Draws the game board and the current state of the board.

    Parameters:
    - board (np.ndarray): 2D array representing the game state.
    """
    screen.fill(GREY)

    # Draw cells
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            cell_rect = pygame.Rect(
                MARGIN + col * (CELL_SIZE + MARGIN), MARGIN + row * (CELL_SIZE + MARGIN), CELL_SIZE, CELL_SIZE
            )
            pygame.draw.rect(screen, WHITE, cell_rect)

            # Draw X or O
            if board[row][col] == 1:
                draw_X(row, col)
            elif board[row][col] == -1:
                draw_O(row, col)

    # Update the display
    pygame.display.flip()


def draw_X(row, col):
    """
    Draws an X in the specified cell.

    Parameters:
    - row (int): Row index.
    - col (int): Column index.
    """
    start_pos1 = (MARGIN + col * (CELL_SIZE + MARGIN) + 20, MARGIN + row * (CELL_SIZE + MARGIN) + 20)
    end_pos1 = (MARGIN + (col + 1) * (CELL_SIZE + MARGIN) - 20, MARGIN + (row + 1) * (CELL_SIZE + MARGIN) - 20)
    start_pos2 = (MARGIN + (col + 1) * (CELL_SIZE + MARGIN) - 20, MARGIN + row * (CELL_SIZE + MARGIN) + 20)
    end_pos2 = (MARGIN + col * (CELL_SIZE + MARGIN) + 20, MARGIN + (row + 1) * (CELL_SIZE + MARGIN) - 20)
    pygame.draw.line(screen, RED, start_pos1, end_pos1, 5)
    pygame.draw.line(screen, RED, start_pos2, end_pos2, 5)


def draw_O(row, col):
    """
    Draws an O in the specified cell.

    Parameters:
    - row (int): Row index.
    - col (int): Column index.
    """
    center = (
        MARGIN + col * (CELL_SIZE + MARGIN) + CELL_SIZE // 2,
        MARGIN + row * (CELL_SIZE + MARGIN) + CELL_SIZE // 2,
    )
    pygame.draw.circle(screen, BLUE, center, CELL_SIZE // 2 - 20, 5)


def display_message(message, color=BLACK):
    """
    Displays a message at the bottom of the window.

    Parameters:
    - message (str): The message to display.
    - color (tuple): RGB color of the text.
    """
    text = FONT.render(message, True, color)
    text_rect = text.get_rect(center=(WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] - 50))
    screen.blit(text, text_rect)
    pygame.display.flip()


def display_end_message(message, color=BLACK):
    """
    Displays the end game message prominently.

    Parameters:
    - message (str): The message to display.
    - color (tuple): RGB color of the text.
    """
    text = END_FONT.render(message, True, color)
    text_rect = text.get_rect(center=(WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2))
    screen.blit(text, text_rect)
    pygame.display.flip()


def get_cell_from_mouse(pos):
    """
    Converts mouse position to board coordinates.

    Parameters:
    - pos (tuple): (x, y) mouse position.

    Returns:
    - (int, int): (row, col) indices on the board.
    """
    x, y = pos
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            cell_rect = pygame.Rect(
                MARGIN + col * (CELL_SIZE + MARGIN), MARGIN + row * (CELL_SIZE + MARGIN), CELL_SIZE, CELL_SIZE
            )
            if cell_rect.collidepoint(x, y):
                return row, col
    return None, None


def ai_move(state):
    """
    Uses MCTS to determine the AI's move.

    Parameters:
    - state (TicTacToeGameState): Current game state.

    Returns:
    - TicTacToeMove: The chosen move by AI.
    """
    root_node = TwoPlayersGameMonteCarloTreeSearchNode(state=state)
    mcts = MonteCarloTreeSearch(root_node, num_processes=4)
    best_node = mcts.best_action_parallel(simulations_number=1000)
    if best_node and best_node.parent_action:
        return best_node.parent_action
    return None


def play_game_pygame(board_size: int = 5, connect: int = 4, simulations_per_move: int = 10000):
    # Initialize the game state
    initial_board_state = np.zeros((board_size, board_size), dtype=int)
    state = TicTacToeGameState(state=initial_board_state, next_to_move=0, win=connect)

    game_over = False
    winner = None

    while True:
        draw_board(state.board)

        if game_over:
            if winner == 1:
                display_end_message("Player X Wins!", RED)
            elif winner == -1:
                display_end_message("Player O Wins!", BLUE)
            else:
                display_end_message("It's a Draw!", BLACK)
        else:
            if state.next_to_move == 0:
                display_message("Your Turn (X)", RED)
            else:
                display_message("AI's Turn (O)", BLUE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                if state.next_to_move == 0:
                    pos = pygame.mouse.get_pos()
                    row, col = get_cell_from_mouse(pos)
                    if row is not None and col is not None:
                        if state.board[row][col] == 0:
                            move = TicTacToeMove(x_coordinate=row, y_coordinate=col, player=0)
                            if state.is_move_legal(move):
                                state = state.move(move)
                                # Check for game over
                                if state.is_game_over():
                                    game_over = True
                                    winner = state.game_result
                                else:
                                    # AI's turn
                                    display_message("AI is thinking...", BLUE)
                                    pygame.display.flip()
                                    pygame.event.pump()  # Allow pygame to process internal actions

                                    # To prevent the UI from freezing, run AI move after a short delay
                                    pygame.time.delay(100)  # 100 milliseconds

                                    ai_action = ai_move(state)
                                    if ai_action:
                                        state = state.move(ai_action)
                                        if state.is_game_over():
                                            game_over = True
                                            winner = state.game_result
                                    else:
                                        # No valid AI moves
                                        game_over = True
                                        winner = 0
        # Control the frame rate
        clock.tick(30)


if __name__ == "__main__":
    play_game_pygame(board_size=5, connect=4, simulations_per_move=10000)
