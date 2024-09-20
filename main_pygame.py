# main_pygame.py

import numpy as np
import sys
import time
try:
    import pygame
except ImportError:
    raise ImportError("Please install pygame using 'pip install pygame' before running this script.")

from pymcts.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from pymcts.tree.search import MonteCarloTreeSearch
from pymcts.games.tictactoe2 import TicTacToeGameState, TicTacToeMove

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
WHITE = (250, 250, 250)
BLACK = (10, 10, 10)
GREY = (50, 50, 50)
RED = (240, 30, 0)
BLUE = (0, 120, 255)
YELLOW = (240, 240, 0)
GREEN = (30, 240, 0)

# Fonts
FONT = pygame.font.SysFont("Arial", 40)
END_FONT = pygame.font.SysFont("Arial", 60)

# Initialize the display
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("MCTS Tic-Tac-Toe")

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Create X and O images programmatically
X_IMAGE = pygame.Surface((CELL_SIZE - 40, CELL_SIZE - 40), pygame.SRCALPHA)
O_IMAGE = pygame.Surface((CELL_SIZE - 40, CELL_SIZE - 40), pygame.SRCALPHA)

# Draw X on X_IMAGE surface
pygame.draw.line(X_IMAGE, RED, (0, 0), (CELL_SIZE - 40, CELL_SIZE - 40), 8)
pygame.draw.line(X_IMAGE, RED, (CELL_SIZE - 40, 0), (0, CELL_SIZE - 40), 8)

# Draw O on O_IMAGE surface
pygame.draw.circle(O_IMAGE, BLUE, ((CELL_SIZE - 40) // 2, (CELL_SIZE - 40) // 2), (CELL_SIZE - 40) // 2 - 5, 8)


def draw_gradient_background(screen, color_start, color_end):
    """Draws a vertical gradient background."""
    for y in range(WINDOW_SIZE[1]):
        ratio = y / WINDOW_SIZE[1]
        color = (
            int(color_start[0] * (1 - ratio) + color_end[0] * ratio),
            int(color_start[1] * (1 - ratio) + color_end[1] * ratio),
            int(color_start[2] * (1 - ratio) + color_end[2] * ratio),
        )
        pygame.draw.line(screen, color, (0, y), (WINDOW_SIZE[0], y))


def draw_board(board, last_move=None, winning_sequence=None):
    # Draw gradient background
    draw_gradient_background(screen, GREY, WHITE)

    # Draw cells
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            cell_rect = pygame.Rect(
                MARGIN + col * (CELL_SIZE + MARGIN), MARGIN + row * (CELL_SIZE + MARGIN), CELL_SIZE, CELL_SIZE
            )
            pygame.draw.rect(screen, WHITE, cell_rect)
            pygame.draw.rect(screen, BLACK, cell_rect, 1)
            # Draw X or O
            if board[row][col] == 1:
                draw_X(row, col)
            elif board[row][col] == -1:
                draw_O(row, col)

    # Highlight the last move
    if last_move:
        row, col = last_move
        highlight_rect = pygame.Rect(
            MARGIN + col * (CELL_SIZE + MARGIN), MARGIN + row * (CELL_SIZE + MARGIN), CELL_SIZE, CELL_SIZE
        )
        pygame.draw.rect(screen, YELLOW, highlight_rect, 5)  # Yellow border

    # Highlight winning sequence
    if winning_sequence:
        for row, col in winning_sequence:
            highlight_rect = pygame.Rect(
                MARGIN + col * (CELL_SIZE + MARGIN), MARGIN + row * (CELL_SIZE + MARGIN), CELL_SIZE, CELL_SIZE
            )
            pygame.draw.rect(screen, GREEN, highlight_rect, 5)  # Green border


def draw_X(row, col):
    position = (MARGIN + col * (CELL_SIZE + MARGIN) + 20, MARGIN + row * (CELL_SIZE + MARGIN) + 20)
    screen.blit(X_IMAGE, position)


def draw_O(row, col):
    position = (MARGIN + col * (CELL_SIZE + MARGIN) + 20, MARGIN + row * (CELL_SIZE + MARGIN) + 20)
    screen.blit(O_IMAGE, position)


def display_message(message, color=BLACK):
    # Draw background rectangle
    bg_rect = pygame.Rect(0, WINDOW_SIZE[1] - 100, WINDOW_SIZE[0], 100)
    pygame.draw.rect(screen, GREY, bg_rect)
    # Render and blit the text
    text = FONT.render(message, True, color)
    text_rect = text.get_rect(center=(WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] - 50))
    screen.blit(text, text_rect)


def display_end_message(message, color=BLACK):
    # Draw semi-transparent overlay
    overlay = pygame.Surface((WINDOW_SIZE[0], WINDOW_SIZE[1]), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))  # Semi-transparent black
    screen.blit(overlay, (0, 0))
    # Render and blit the text
    text = END_FONT.render(message, True, color)
    text_rect = text.get_rect(center=(WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2))
    screen.blit(text, text_rect)


def get_cell_from_mouse(pos):
    x, y = pos
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            cell_rect = pygame.Rect(
                MARGIN + col * (CELL_SIZE + MARGIN), MARGIN + row * (CELL_SIZE + MARGIN), CELL_SIZE, CELL_SIZE
            )
            if cell_rect.collidepoint(x, y):
                return row, col
    return None, None


def ai_move(state, simulations_per_move):
    root_node = TwoPlayersGameMonteCarloTreeSearchNode(state=state)
    mcts = MonteCarloTreeSearch(root_node, num_processes=4)
    best_node = mcts.best_action_parallel(simulations_number=simulations_per_move)
    if best_node and best_node.parent_action:
        return best_node.parent_action
    return None


def play_game_pygame(board_size: int = 5, connect: int = 4, simulations_per_move: int = 10000):
    initial_board_state = np.zeros((board_size, board_size), dtype=int)
    state = TicTacToeGameState(state=initial_board_state, next_to_move=0, win=connect)
    game_over = False
    winner = None
    last_move = None
    winning_sequence = None

    ai_needs_to_move = False  # Flag to indicate if AI needs to make a move

    while True:
        # Draw the game state
        draw_board(state.board, last_move=last_move, winning_sequence=winning_sequence if game_over else None)

        # Display messages
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
                display_message("AI is thinking...", BLUE)

        # Update the display once per frame
        pygame.display.flip()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                if state.next_to_move == 0:
                    pos = pygame.mouse.get_pos()
                    row, col = get_cell_from_mouse(pos)
                    if row is not None and col is not None and state.board[row][col] == 0:
                        move = TicTacToeMove(x_coordinate=row, y_coordinate=col, player=0)
                        if state.is_move_legal(move):
                            state = state.move(move)
                            last_move = (move.x_coordinate, move.y_coordinate)
                            # Update the display to show the player's move
                            draw_board(state.board, last_move=last_move)
                            display_message("AI is thinking...", BLUE)
                            pygame.display.flip()
                            # Check for game over
                            if state.is_game_over():
                                game_over = True
                                winner = state.game_result
                                winning_sequence = state.winning_sequence
                            else:
                                # Set flag to indicate AI needs to move
                                ai_needs_to_move = True

        # AI's turn outside of event loop
        if ai_needs_to_move and not game_over:
            # AI's turn
            ai_action = ai_move(state, simulations_per_move)
            if ai_action:
                state = state.move(ai_action)
                last_move = (ai_action.x_coordinate, ai_action.y_coordinate)
                if state.is_game_over():
                    game_over = True
                    winner = state.game_result
                    winning_sequence = state.winning_sequence
            else:
                # No valid AI moves
                game_over = True
                winner = 0
                winning_sequence = None
            ai_needs_to_move = False  # Reset the flag

        # Control the frame rate
        clock.tick(30)


if __name__ == "__main__":
    play_game_pygame(board_size=5, connect=4, simulations_per_move=20000)
