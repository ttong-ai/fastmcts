import numpy as np
import queue
import sys
import threading
import time

try:
    import pygame
except ImportError:
    raise ImportError("Please install pygame using 'pip install pygame' before running this script.")

from fastmcts.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from fastmcts.tree.search import MonteCarloTreeSearch
from fastmcts.games.tictactoe2 import TicTacToeGameState, TicTacToeMove

# Initialize Pygame
pygame.init()

# Colors
WHITE = (250, 250, 250)
BLACK = (10, 10, 10)
GREY = (50, 50, 50)
RED = (240, 30, 0)
BLUE = (0, 120, 255)
YELLOW = (240, 240, 0)
GREEN = (30, 240, 0)

# Fonts
FONT = pygame.font.SysFont("Comic Sans MS", 40)
END_FONT = pygame.font.SysFont("Arial", 60)


def calculate_dimensions(board_size):
    CELL_SIZE = min(100, 500 // board_size)  # Adjust cell size based on board size
    MARGIN = max(2, 5 - board_size // 5)  # Reduce margin for larger boards
    WINDOW_SIZE = (
        board_size * CELL_SIZE + (board_size + 1) * MARGIN,
        board_size * CELL_SIZE + (board_size + 1) * MARGIN + 100,
    )
    return CELL_SIZE, MARGIN, WINDOW_SIZE


def initialize_pygame(board_size):
    CELL_SIZE, MARGIN, WINDOW_SIZE = calculate_dimensions(board_size)

    # Initialize the display
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("MCTS Tic-Tac-Toe")

    # Create X and O images programmatically
    X_IMAGE = pygame.Surface((CELL_SIZE - 20, CELL_SIZE - 20), pygame.SRCALPHA)
    O_IMAGE = pygame.Surface((CELL_SIZE - 20, CELL_SIZE - 20), pygame.SRCALPHA)

    # Calculate the thickness
    thickness = max(5, (CELL_SIZE - 20) // 8)

    # Calculate the size reduction for X
    x_size_reduction = (CELL_SIZE - 20) // 5

    # Draw X on X_IMAGE surface (thicker and slightly smaller)
    pygame.draw.line(X_IMAGE, RED, (x_size_reduction, x_size_reduction),
                     (CELL_SIZE - 20 - x_size_reduction, CELL_SIZE - 20 - x_size_reduction), thickness)
    pygame.draw.line(X_IMAGE, RED, (CELL_SIZE - 20 - x_size_reduction, x_size_reduction),
                     (x_size_reduction, CELL_SIZE - 20 - x_size_reduction), thickness)

    # Draw O on O_IMAGE surface
    pygame.draw.circle(
        O_IMAGE,
        BLUE,
        ((CELL_SIZE - 20) // 2, (CELL_SIZE - 20) // 2),
        (CELL_SIZE - 20) // 2 - thickness // 2,
        thickness,
    )

    return screen, CELL_SIZE, MARGIN, WINDOW_SIZE, X_IMAGE, O_IMAGE


def draw_gradient_background(screen, color_start, color_end, WINDOW_SIZE):
    """Draws a vertical gradient background."""
    for y in range(WINDOW_SIZE[1]):
        ratio = y / WINDOW_SIZE[1]
        color = (
            int(color_start[0] * (1 - ratio) + color_end[0] * ratio),
            int(color_start[1] * (1 - ratio) + color_end[1] * ratio),
            int(color_start[2] * (1 - ratio) + color_end[2] * ratio),
        )
        pygame.draw.line(screen, color, (0, y), (WINDOW_SIZE[0], y))


def draw_board(screen, board, CELL_SIZE, MARGIN, X_IMAGE, O_IMAGE, last_move=None, winning_sequence=None):
    # Draw gradient background
    draw_gradient_background(screen, GREY, WHITE, (screen.get_width(), screen.get_height()))

    # Draw cells
    for row in range(len(board)):
        for col in range(len(board)):
            cell_rect = pygame.Rect(
                MARGIN + col * (CELL_SIZE + MARGIN), MARGIN + row * (CELL_SIZE + MARGIN), CELL_SIZE, CELL_SIZE
            )
            pygame.draw.rect(screen, WHITE, cell_rect)
            pygame.draw.rect(screen, BLACK, cell_rect, 1)
            # Draw X or O
            if board[row][col] == 1:
                screen.blit(X_IMAGE, (cell_rect.x + 10, cell_rect.y + 10))
            elif board[row][col] == -1:
                screen.blit(O_IMAGE, (cell_rect.x + 10, cell_rect.y + 10))

    # Highlight the last move
    if last_move:
        row, col = last_move
        highlight_rect = pygame.Rect(
            MARGIN + col * (CELL_SIZE + MARGIN), MARGIN + row * (CELL_SIZE + MARGIN), CELL_SIZE, CELL_SIZE
        )
        pygame.draw.rect(screen, YELLOW, highlight_rect, 3)  # Yellow border

    # Highlight winning sequence
    if winning_sequence:
        for row, col in winning_sequence:
            highlight_rect = pygame.Rect(
                MARGIN + col * (CELL_SIZE + MARGIN), MARGIN + row * (CELL_SIZE + MARGIN), CELL_SIZE, CELL_SIZE
            )
            pygame.draw.rect(screen, GREEN, highlight_rect, 3)  # Green border


def display_message(screen, message, color=BLACK):
    # Draw background rectangle
    bg_rect = pygame.Rect(0, screen.get_height() - 100, screen.get_width(), 100)
    pygame.draw.rect(screen, GREY, bg_rect)
    # Render and blit the text
    text = FONT.render(message, True, color)
    text_rect = text.get_rect(center=(screen.get_width() // 2, screen.get_height() - 50))
    screen.blit(text, text_rect)


def display_end_message(screen, message, color=BLACK):
    # Draw semi-transparent overlay
    overlay = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))  # Semi-transparent black
    screen.blit(overlay, (0, 0))
    # Render and blit the text
    text = END_FONT.render(message, True, color)
    text_rect = text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
    screen.blit(text, text_rect)


def get_cell_from_mouse(pos, board_size, CELL_SIZE, MARGIN):
    x, y = pos
    for row in range(board_size):
        for col in range(board_size):
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


def compute_ai_move(state, simulations_per_move, move_queue):
    """Function to run AI move computation in a separate thread."""
    move = ai_move(state, simulations_per_move)
    move_queue.put(move)


def play_game_pygame(board_size: int = 5, connect: int = 4, simulations_per_move: int = 10000):
    screen, CELL_SIZE, MARGIN, WINDOW_SIZE, X_IMAGE, O_IMAGE = initialize_pygame(board_size)

    initial_board_state = np.zeros((board_size, board_size), dtype=int)
    state = TicTacToeGameState(state=initial_board_state, next_to_move=0, win=connect)
    game_over = False
    winner = None
    last_move = None
    winning_sequence = None

    ai_needs_to_move = False
    ai_thread = None
    ai_move_queue = queue.Queue()

    clock = pygame.time.Clock()

    while True:
        # Draw the game state
        draw_board(
            screen,
            state.board,
            CELL_SIZE,
            MARGIN,
            X_IMAGE,
            O_IMAGE,
            last_move,
            winning_sequence if game_over else None,
        )

        # Display messages
        if game_over:
            if winner == 1:
                display_end_message(screen, "Player X Wins!", RED)
            elif winner == -1:
                display_end_message(screen, "Player O Wins!", BLUE)
            else:
                display_end_message(screen, "It's a Draw!", BLACK)
        else:
            if state.next_to_move == 0:
                display_message(screen, "Your Turn (X)", RED)
            else:
                display_message(screen, "AI is thinking...", BLUE)

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
                    row, col = get_cell_from_mouse(pos, board_size, CELL_SIZE, MARGIN)
                    if row is not None and col is not None and state.board[row][col] == 0:
                        move = TicTacToeMove(x_coordinate=row, y_coordinate=col, player=0)
                        if state.is_move_legal(move):
                            state = state.move(move)
                            last_move = (move.x_coordinate, move.y_coordinate)
                            # Update the display to show the player's move
                            draw_board(screen, state.board, CELL_SIZE, MARGIN, X_IMAGE, O_IMAGE, last_move)
                            display_message(screen, "AI is thinking...", BLUE)
                            pygame.display.flip()
                            # Check for game over
                            if state.is_game_over():
                                game_over = True
                                winner = state.game_result
                                winning_sequence = state.winning_sequence
                            else:
                                # Set flag to indicate AI needs to move
                                ai_needs_to_move = True

        # Check if AI needs to move and is not already computing
        if ai_needs_to_move and ai_thread is None:
            # Start the AI computation in a new thread
            ai_thread = threading.Thread(target=compute_ai_move, args=(state, simulations_per_move, ai_move_queue))
            ai_thread.start()

        # Check if AI has computed its move
        if ai_thread is not None:
            try:
                # Non-blocking check for AI move
                ai_action = ai_move_queue.get_nowait()
            except queue.Empty:
                # AI is still computing
                ai_action = None

            if ai_action is not None:
                # AI has finished computing its move
                state = state.move(ai_action)
                last_move = (ai_action.x_coordinate, ai_action.y_coordinate)
                if state.is_game_over():
                    game_over = True
                    winner = state.game_result
                    winning_sequence = state.winning_sequence
                # Reset flags and thread reference
                ai_needs_to_move = False
                ai_thread = None

        # Control the frame rate
        clock.tick(30)


if __name__ == "__main__":
    play_game_pygame(board_size=7, connect=4, simulations_per_move=20000)
