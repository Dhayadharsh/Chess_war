#this is comment
import pygame
import numpy as np

BOARD_SIZE = 16
SQUARE_SIZE = 50
BOARD_COLS = 16
BOARD_ROWS = 16
WINDOW_SIZE = BOARD_SIZE * SQUARE_SIZE
WINDOW_SIZE = (BOARD_COLS * SQUARE_SIZE, BOARD_ROWS * SQUARE_SIZE)
GRAY=(25,45,45)
PWHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BOARD_GREEN = (153, 255, 153)
BOARD_WHITE = (200, 200, 200)
HIGHLIGHT_COLOR = (255, 255, 0)
CHECK_COLOR = (255, 0, 0)
MIDLINE_color = (255, 255, 255)
FONT_COLOR = BLACK
SIDEBAR_WIDTH = 200
# Initialize Pygame
pygame.init()


# Piece class to represent chess pieces
class Piece:
    def __init__(self, symbol, color):
        self.symbol = symbol
        self.color = color

    def draw(self, screen, row, col):
        font = pygame.font.SysFont(None, SQUARE_SIZE)
        img = font.render(self.symbol, True, self.color)
        screen.blit(img, (col * SQUARE_SIZE + SQUARE_SIZE // 4, row * SQUARE_SIZE + SQUARE_SIZE // 4))

    def get_valid_moves(self, board, row, col):
        return []


class Queen(Piece):
    def __init__(self, color):
        super().__init__('Q', color)
        self.has_moved = True

    def get_valid_moves(self, board, row, col):
        valid_moves = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for direction in directions:
            for i in range(1, BOARD_SIZE):
                new_row = row + direction[0] * i
                new_col = col + direction[1] * i
                if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                    if board[new_row][new_col] is None:
                        valid_moves.append((new_row, new_col))
                    elif board[new_row][new_col].color != self.color:
                        valid_moves.append((new_row, new_col))
                        break
                    else:
                        break
                else:
                    break
        return valid_moves

    def get_movement_moves(self, board, row, col):
        return self.get_valid_moves(board, row, col)


# Queen class to represent queens with movement rules
class Rook(Piece):
    def __init__(self, color):
        super().__init__('R', color)
        self.has_moved = True

    def get_valid_moves(self, board, row, col):
        valid_moves = []

        # Define movement directions for the queen
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # Vertical and horizontal

        ]

        for direction in directions:
            for i in range(1, BOARD_SIZE):
                new_row = row + direction[0] * i
                new_col = col + direction[1] * i
                if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                    if board[new_row][new_col] is None:
                        valid_moves.append((new_row, new_col))
                    elif board[new_row][new_col].color != self.color:
                        valid_moves.append((new_row, new_col))
                        break
                    else:
                        break
                else:
                    break

        return valid_moves

    def get_movement_moves(self, board, row, col):
        return self.get_valid_moves(board, row, col)


# King class to represent kings with movement rules
class King(Piece):
    def __init__(self, color):
        super().__init__('K', color)
        self.has_moved = True

    def get_valid_moves(self, board, row, col):
        valid_moves = []

        # Define movement directions for the king
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # Vertical and horizontal
            (1, 1), (-1, -1), (1, -1), (-1, 1)  # Diagonal
        ]

        for direction in directions:
            new_row = row + direction[0]
            new_col = col + direction[1]
            if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                if board[new_row][new_col] is None or board[new_row][new_col].color != self.color:
                    valid_moves.append((new_row, new_col))

        return valid_moves

    def get_movement_moves(self, board, row, col):
        return self.get_valid_moves(board, row, col)


class Bishop(Piece):
    def __init__(self, color):
        super().__init__('B', color)
        self.has_moved = True

    def get_valid_moves(self, board, row, col):
        valid_moves = []
        directions = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
        for direction in directions:
            for i in range(1, BOARD_SIZE):
                new_row = row + direction[0] * i
                new_col = col + direction[1] * i
                if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                    if board[new_row][new_col] is None:
                        valid_moves.append((new_row, new_col))
                    elif board[new_row][new_col].color != self.color:
                        valid_moves.append((new_row, new_col))
                        break
                    else:
                        break
                else:
                    break
        return valid_moves

    def get_movement_moves(self, board, row, col):
        return self.get_valid_moves(board, row, col)


class swordsman(Piece):
    def __init__(self, color):
        super().__init__('S', color)
        self.has_moved = True

    def get_movement_moves(self, board, row, col):
        valid_moves = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for direction in directions:

            new_row = row + direction[0] * 2
            new_col = col + direction[1] * 2

            if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:

                if board[new_row - direction[0] * (direction[0] != 0)][
                    new_col - direction[1] * (direction[1] != 0)] == None and board[new_row][new_col] is None:
                    valid_moves.append((new_row, new_col))
        return valid_moves

    def get_valid_moves(self, board, row, col):
        valid_moves = []

        # Define movement directions for the king
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # Vertical and horizontal
            (1, 1), (-1, -1), (1, -1), (-1, 1)  # Diagonal
        ]

        for direction in directions:
            new_row = row + direction[0]
            new_col = col + direction[1]
            if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                if board[new_row][new_col] is not None and board[new_row][new_col].color != self.color:
                    valid_moves.append((new_row, new_col))

        return valid_moves


class Knight(Piece):
    def __init__(self, color):
        super().__init__('N', color)
        self.has_moved = True

    def get_valid_moves(self, board, row, col):
        valid_moves = []
        moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
        for move in moves:
            new_row = row + move[0]
            new_col = col + move[1]
            if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                if board[new_row][new_col] is None or board[new_row][new_col].color != self.color:
                    valid_moves.append((new_row, new_col))
        return valid_moves

    def get_movement_moves(self, board, row, col):
        return self.get_valid_moves(board, row, col)


# Queen class to represent bishop with movement rules


# Pawn class to represent pawns with movement rules
class Pawn(Piece):
    def __init__(self, color, ser_num):
        super().__init__('P', color)
        if color == PWHITE:
            self.direction = -1
        else:
            self.direction = 1  # Direction can be 1 for moving downwards or -1 for moving upwards
        self.has_moved = False

    def get_valid_moves(self, board, row, col):
        valid_moves = []

        # Forward move by 1
        if 0 <= row + self.direction < BOARD_SIZE and board[row + self.direction][col] is None:
            valid_moves.append((row + self.direction, col))

            # Initial double move by 2
            if not self.has_moved and 0 <= row + 2 * self.direction < BOARD_SIZE and board[row + 2 * self.direction][
                col] is None:
                valid_moves.append((row + 2 * self.direction, col))

        # Diagonal captures
        for offset in [-1, 1]:
            new_col = col + offset
            if 0 <= row + self.direction < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                target = board[row + self.direction][new_col]
                if target is not None and target.color != self.color:
                    valid_moves.append((row + self.direction, new_col))

        return valid_moves

    def get_movement_moves(self, board, row, col):
        return self.get_valid_moves(board, row, col)


pk = 0


def symbol_to_object(symbol, color):
    if symbol == 'R':
        return Rook(color)
    if symbol == 'K':
        return King(color)
    if symbol == 'B':
        return Bishop(color)
    if symbol == 'S':
        return swordsman(color)
    if symbol == 'N':
        return Knight(color)
    if symbol == 'Q':
        return Queen(color)
    if symbol == 'P':
        global pk
        pk = pk + 1
        return Pawn(color, pk)
    if symbol == "":
        return None


def object_to_symbol(piece):
    if isinstance(piece, Rook):
        return 'R'
    elif isinstance(piece, King):
        return 'K'
    elif isinstance(piece, Bishop):
        return 'B'
    elif isinstance(piece, swordsman):
        return 'S'
    elif isinstance(piece, Knight):
        return 'N'
    elif isinstance(piece, Queen):
        return 'Q'
    elif isinstance(piece, Pawn):
        return 'P'
    elif piece == None:
        return ""
    else:
        raise ValueError("Invalid chess piece object")


def adjacent_pos(position):
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),  # Vertical and horizontal
        (1, 1), (-1, -1), (1, -1), (-1, 1)  # Diagonal
    ]
    adjacent_square = []
    for direction in directions:
        new_row = position[0] + direction[0]
        new_col = position[1] + direction[1]
        if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
            adjacent_square.append((new_row, new_col))
    return adjacent_square


class ChessBoard:
    def __init__(self):
        self.board = np.full((BOARD_SIZE, BOARD_SIZE), None)
        self.king_positions = {'white': None, 'black': None}

    def place_piece(self, piece, position):
        row, col = position
        self.board[row][col] = piece
        if isinstance(piece, King):
            self.king_positions['white' if piece.color == PWHITE else 'black'] = (row, col)

    def remove_piece(self, position):
        row, col = position
        self.board[row][col] = None

    def move_piece(self, start_pos, end_pos):
        piece = self.board[start_pos[0]][start_pos[1]]
        if piece:
            piece.has_moved = True
            self.board[end_pos[0]][end_pos[1]] = piece
            self.board[start_pos[0]][start_pos[1]] = None
            if isinstance(piece, King):
                self.king_positions['white' if piece.color == PWHITE else 'black'] = end_pos

    def is_in_check(self, color):
        king_pos = self.king_positions['white' if color == PWHITE else 'black']
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.board[row][col]
                if piece and piece.color != color:
                    if king_pos in piece.get_valid_moves(self.board, row, col):
                        return True
        return False

    def get_valid_moves(self, piece, row, col):
        potential_moves = piece.get_movement_moves(self.board, row, col)
        valid_moves = []

        for move in potential_moves:
            if not self.move_puts_king_in_check(piece, row, col, move):
                valid_moves.append(move)
        return valid_moves

    def swordkill(self, piece, row, col):
        potential_moves = piece.get_valid_moves(self.board, row, col)
        valid_moves = []
        for move in potential_moves:
            valid_moves.append(move)
        return valid_moves

    def move_puts_king_in_check(self, piece, start_row, start_col, move):
        # Temporarily make the move
        end_row, end_col = move
        original_piece = self.board[end_row][end_col]
        self.board[end_row][end_col] = piece
        self.board[start_row][start_col] = None

        if isinstance(piece, King):
            self.king_positions['white' if piece.color == PWHITE else 'black'] = (end_row, end_col)

        in_check = self.is_in_check(piece.color)

        # Undo the move
        self.board[start_row][start_col] = piece
        self.board[end_row][end_col] = original_piece

        if isinstance(piece, King):
            self.king_positions['white' if piece.color == PWHITE else 'black'] = (start_row, start_col)

        return in_check

    def draw(self, screen, selected_piece, valid_moves, movement_moves, group_move, p):
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                color = BOARD_WHITE if (row + col) % 2 == 0 else BOARD_GREEN
                pygame.draw.rect(screen, color,
                                 pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

                piece = self.board[row][col]
                if piece is not None:
                    piece.draw(screen, row, col)
                if p:
                    if (row, col) in group_move:
                        pygame.draw.rect(screen, HIGHLIGHT_COLOR,
                                         pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 5)
                else:
                    if (row, col) in valid_moves or (row, col) in movement_moves:
                        pygame.draw.rect(screen, HIGHLIGHT_COLOR,
                                         pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 5)

        if self.is_in_check(PWHITE):
            king_pos = self.king_positions['white']
            pygame.draw.rect(screen, CHECK_COLOR,
                             pygame.Rect(king_pos[1] * SQUARE_SIZE, king_pos[0] * SQUARE_SIZE, SQUARE_SIZE,
                                         SQUARE_SIZE), 5)

        if self.is_in_check(BLACK):
            king_pos = self.king_positions['black']
            pygame.draw.rect(screen, CHECK_COLOR,
                             pygame.Rect(king_pos[1] * SQUARE_SIZE, king_pos[0] * SQUARE_SIZE, SQUARE_SIZE,
                                         SQUARE_SIZE), 5)

    def plaindraw(self, screen):
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                color = BOARD_GREEN if (row + col) % 2 == 0 else BOARD_WHITE
                pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                piece = self.board[row][col]
                if piece:
                    piece.draw(screen, row, col)


def draw_midline(screen):
    for col in range(BOARD_COLS):
        pygame.draw.line(screen, MIDLINE_color, (col * SQUARE_SIZE, BOARD_ROWS // 2 * SQUARE_SIZE),
                         ((col + 1) * SQUARE_SIZE, BOARD_ROWS // 2 * SQUARE_SIZE), 2)


import os
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox

BOARD_COLS = BOARD_SIZE
BOARD_ROWS = BOARD_SIZE


def draw_sidebar(screen, selected_piece, counts):
    sidebar_rect = pygame.Rect(BOARD_COLS * SQUARE_SIZE, 0, SIDEBAR_WIDTH, BOARD_ROWS * SQUARE_SIZE)
    pygame.draw.rect(screen, BLACK, sidebar_rect)

    pieces = ['K', 'N', 'S', 'P', 'B', 'Q', 'R']
    piece_names = ['King', 'Knight', 'Swordsman', 'Pawn', 'Bishop', 'Queen', 'Rook']
    for i, (piece, count) in enumerate(zip(pieces, counts)):
        font = pygame.font.SysFont(None, SQUARE_SIZE)
        img = font.render(f"{piece} {count}", True, PWHITE)
        screen.blit(img, (BOARD_COLS * SQUARE_SIZE + 20, i * SQUARE_SIZE + 20))

    if selected_piece:
        img = font.render(selected_piece.symbol, True, HIGHLIGHT_COLOR)
        screen.blit(img, (BOARD_COLS * SQUARE_SIZE + 20, pieces.index(selected_piece.symbol) * SQUARE_SIZE + 20))


def white_placer():
    screen = pygame.display.set_mode((WINDOW_SIZE[0] + SIDEBAR_WIDTH, WINDOW_SIZE[1]))
    pygame.display.set_caption("White Player Placement")
    board = ChessBoard()
    selected_piece = None
    running = True

    # Piece counts
    pc, PC = 0, 32
    kc, KC = 0, 1
    qc, QC = 0, 4
    oc, OC = 0, 16
    rc, RC = 0, 8
    bc, nc, sc = 0, 0, 0
    piece_counts = [kc, nc, sc, pc, bc, qc, rc]  # Corresponding piece counts for King, Knight, Swordsman, Pawn, etc.

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if x >= BOARD_COLS * SQUARE_SIZE:
                    selected_piece = \
                    [King(PWHITE), Knight(PWHITE), swordsman(PWHITE), Pawn(PWHITE, pc), Bishop(PWHITE), Queen(PWHITE),
                     Rook(PWHITE)][(y // SQUARE_SIZE) % 7]
                elif y // SQUARE_SIZE > 8:
                    row, col = y // SQUARE_SIZE, x // SQUARE_SIZE
                    if selected_piece and event.button == 1:
                        if isinstance(selected_piece, Pawn) and pc < PC:
                            board.place_piece(Pawn(PWHITE, pc), (row, col))
                            pc += 1
                            piece_counts[3] = pc
                        elif isinstance(selected_piece, King) and kc < KC:
                            board.place_piece(King(PWHITE), (row, col))
                            kc += 1
                            piece_counts[0] = kc
                        elif isinstance(selected_piece, Queen) and qc < QC:
                            board.place_piece(Queen(PWHITE), (row, col))
                            qc += 1
                            piece_counts[5] = qc
                        elif isinstance(selected_piece, Rook) and rc < RC:
                            board.place_piece(Rook(PWHITE), (row, col))
                            rc += 1
                            piece_counts[6] = rc
                        elif isinstance(selected_piece, Bishop) or isinstance(selected_piece, Knight) or isinstance(
                                selected_piece, swordsman) and oc < OC:
                            board.place_piece(selected_piece, (row, col))
                            if isinstance(selected_piece, Bishop):
                                bc = bc + 1
                                piece_counts[4] = bc
                            if isinstance(selected_piece, swordsman):
                                sc = sc + 1
                                piece_counts[2] = sc
                            if isinstance(selected_piece, Knight):
                                nc = nc + 1
                                piece_counts[1] = nc

                            oc += 1

                    elif selected_piece and event.button == 3:
                        board.remove_piece((row, col))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_k:
                    selected_piece = King(PWHITE)
                elif event.key == pygame.K_n:
                    selected_piece = Knight(PWHITE)
                elif event.key == pygame.K_s:
                    selected_piece = swordsman(PWHITE)
                elif event.key == pygame.K_p:
                    selected_piece = Pawn(PWHITE, pc)
                elif event.key == pygame.K_b:
                    selected_piece = Bishop(PWHITE)
                elif event.key == pygame.K_q:
                    selected_piece = Queen(PWHITE)
                elif event.key == pygame.K_r:
                    selected_piece = Rook(PWHITE)
                if event.key == pygame.K_RETURN:
                    save_prompt = messagebox.askyesno("Save Board", "Do you want to save this board position?")
                    rows, cols = BOARD_ROWS, BOARD_COLS
                    saved_board = [[0 for _ in range(cols)] for _ in range(rows)]
                    if save_prompt:  # If the user says "Yes"
                        for row in range(BOARD_ROWS):
                            for col in range(BOARD_COLS):
                                saved_board[row][col] = object_to_symbol(board.board[row][col])

                        save_board(saved_board)

                    running = False

        screen.fill(BLACK)
        board.plaindraw(screen)
        draw_sidebar(screen, selected_piece, piece_counts)  # Pass piece counts to sidebar
        draw_midline(screen)
        pygame.display.flip()

    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            board.board[row][col] = object_to_symbol(board.board[row][col])

    return board.board


def load_window():
    """Display window for 'New Game' or 'Load Game' selection."""
    screen = pygame.display.set_mode((WINDOW_SIZE[0] + SIDEBAR_WIDTH, WINDOW_SIZE[1]))
    pygame.display.set_caption("Load or New Game")
    font = pygame.font.SysFont(None, 55)
    running = True

    while running:
        screen.fill(BLACK)

        # Button text rendering
        new_game_text = font.render('New Game', True, PWHITE)
        load_game_text = font.render('Load Game', True, PWHITE)

        # Button positioning
        new_game_rect = pygame.Rect(50, 100, 300, 100)
        load_game_rect = pygame.Rect(50, 250, 300, 100)

        # Draw buttons
        pygame.draw.rect(screen, GRAY, new_game_rect)
        pygame.draw.rect(screen, GRAY, load_game_rect)

        # Display button text
        screen.blit(new_game_text, (new_game_rect.x + 20, new_game_rect.y + 20))
        screen.blit(load_game_text, (load_game_rect.x + 20, load_game_rect.y + 20))

        pygame.display.flip()
        p = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()

                # Handle button clicks
                if new_game_rect.collidepoint(mouse_pos):
                    p = 1

                elif load_game_rect.collidepoint(mouse_pos):

                    p = 2
                if (p != 0):
                    running = False
                    print("hhhhh")
    if (p == 1):

        return "new"

    elif (p == 2):
        pygame.quit()
        return "load"


def load_saved_game():
    """Prompts the user to select a saved board using a dropdown (combobox)."""
    saved_files = [f for f in os.listdir() if f.endswith('.json')]

    # Create a new window for the dropdown
    popup = tk.Tk()  # Use Tk() directly, as we won't use root
    popup.title("Load Board")

    label = tk.Label(popup, text="Select a saved board:")
    label.pack(pady=10)

    # Create a combobox (dropdown) for selecting the saved board
    combobox = ttk.Combobox(popup, values=saved_files)

    if saved_files:
        combobox.set(saved_files[0])  # Set the initial value to the first file
    combobox.pack(pady=10)

    # Variable to store the selected file
    selected_file = None

    # Define what happens when the "Load" button is clicked
    def on_select():
        nonlocal selected_file
        selected_file = combobox.get()  # Get the selected value from the combobox
        if not selected_file:
            messagebox.showwarning("Selection Required", "Please select a board.")
        else:
            popup.destroy()  # Close the popup window

    # Add a "Load" button
    load_button = tk.Button(popup, text="Load", command=on_select)
    load_button.pack(pady=10)

    # Make the popup modal so the user must interact with it
    popup.grab_set()

    # Start the popup window's event loop
    popup.mainloop()

    # Return the selected file
    try:
        with open(selected_file, 'r') as file:
            board_data = json.load(file)
        return board_data  # Return the actual content of the JSON file
    except (FileNotFoundError, json.JSONDecodeError) as e:
        messagebox.showerror("Error", f"Failed to load the file: {e}")
        return None


def save_board(board):
    """Prompts the user for a name and saves the current board state to a file."""
    # Create a Tkinter window for input (hidden)
    root = tk.Tk()
    root.withdraw()

    # Ask the user for a name for the saved board
    save_name = simpledialog.askstring("Save Board", "Enter a name for this board position:")

    if save_name:
        # Construct file path
        file_name = f"{save_name}.json"

        # Save the board state as JSON
        with open(file_name, 'w') as f:
            json.dump(board, f)
        messagebox.showinfo("Save Successful", f"Board saved as '{file_name}'")
    else:
        messagebox.showwarning("Save Failed", "No name entered, board not saved.")


import socket
import pickle
import json
import pygame

# ChessBoard and other necessary classes should be defined or imported here.
hostname = socket.gethostname()
ip_address = "172.23.192.103"

print("IP Address:", ip_address)
IHOST = "172.23.5.46"  # Server IP address (same as the client)
IPORT = 12346


def run_client(HOST=ip_address, PORT=IPORT):
    pygame.init()
    result = load_window()
    if result == 'new':
        wboard = white_placer()
        print(wboard)
        wboard = wboard.tolist()
        print("New game started")
    elif result == 'load':
        # Load a previously saved board state
        loaded_board = load_saved_game()
        if loaded_board:
            print("Loaded saved game")
            wboard = loaded_board
            print(type(wboard))
        else:
            print("No saved game found")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((HOST, PORT))

        # Setup initial board for the client (White)

        board_json = json.dumps(wboard )
        client_socket.sendall(board_json.encode('utf-8'))
        print("sent")

        bboard = None
        combined_board_json = client_socket.recv(4096).decode('utf-8')
        print("recieved")
        bboard = json.loads(combined_board_json)  # Black player's board received

        # Set up the chess board
        chess_board = ChessBoard()
        for row in range(16):
            for col in range(16):
                if wboard[row][col] != '':
                    if isinstance(symbol_to_object(wboard[row][col], PWHITE), King):
                        chess_board.king_positions['white'] = (row, col)
                    chess_board.board[row][col] = symbol_to_object(wboard[row][col], PWHITE)
                if bboard[row][col] != '':
                    if isinstance(symbol_to_object(bboard[row][col], BLACK), King):
                        chess_board.king_positions['black'] = (row, col)
                    chess_board.board[row][col] = symbol_to_object(bboard[row][col], BLACK)

        turn_color = PWHITE
        running = True
        WINDOW_SIZE = BOARD_SIZE * SQUARE_SIZE
        selected_piece = None
        valid_moves = []
        movement_moves = []
        pawn_group = []
        group_move = []

        pygame.init()
        screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("16x16 Chess Board (Client)")

        while running:
            def groupchecker(pos, act_pos, pawn_groups):
                boole = False
                pawn_pos = list(pawn_groups)
                if isinstance(chess_board.board[pos[0]][pos[1]], Pawn):
                    pawn = chess_board.board[pos[0]][pos[1]]
                    act_pawn = chess_board.board[act_pos[0]][act_pos[1]]
                    if pawn.color == turn_color:
                        poscheck = False
                        pawn_pos.append(act_pos)
                        for pos1 in pawn_pos:
                            if pos in adjacent_pos(pos1):
                                poscheck = True
                        if poscheck:
                            if ((pos[0] + pawn.direction, pos[1])) in (
                                    chess_board.get_valid_moves(pawn, pos[0], pos[1])):
                                if ((act_pos[0] + act_pawn.direction, act_pos[1])) in (
                                        chess_board.get_valid_moves(act_pawn, act_pos[0], act_pos[1])):
                                    boole = True
                return boole

            if turn_color == PWHITE:
                print("white move")
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        client_socket.send(pickle.dumps(("QUIT", None)))
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        pos = pygame.mouse.get_pos()
                        row, col = pos[1] // SQUARE_SIZE, pos[0] // SQUARE_SIZE

                        if selected_piece:
                            if isinstance(piece, Pawn) and groupchecker((row, col), selected_piece, pawn_group):
                                pawn = chess_board.board[row][col]
                                act_pawn = chess_board.board[selected_piece[0]][selected_piece[1]]
                                pawn_group.append(selected_piece)
                                pawn_group.append((row, col))
                                group_move.append((selected_piece[0] + act_pawn.direction, selected_piece[1]))
                                group_move.append((row + act_pawn.direction, col))
                            else:
                                if (row, col) in group_move:
                                    for i in range(len(pawn_group)):
                                        chess_board.move_piece(pawn_group[i], group_move[i])
                                    turn_color = BLACK if turn_color == PWHITE else PWHITE
                                    pawn_group = []
                                    group_move = []

                                elif (row, col) in valid_moves:
                                    if not isinstance(piece, swordsman):
                                        chess_board.move_piece(selected_piece, (row, col))
                                        turn_color = BLACK if turn_color == PWHITE else PWHITE
                                    else:
                                        if chess_board.board[row][col] != None:
                                            if chess_board.board[row][col].color != turn_color:
                                                chess_board.board[row][col] = None
                                                turn_color = BLACK
                                elif (row, col) in movement_moves:
                                    chess_board.move_piece(selected_piece, (row, col))
                                    turn_color = BLACK

                                selected_piece = None
                                valid_moves = []
                                pawn_group = []
                                group_move = []
                        else:
                            piece = chess_board.board[row][col]
                            if piece and piece.color == turn_color:
                                selected_piece = (row, col)
                                valid_moves = chess_board.get_valid_moves(piece, row, col)
                                if not isinstance(piece, swordsman):
                                    movement_moves = []
                                else:
                                    movement_moves = valid_moves
                                    valid_moves = chess_board.swordkill(piece, row, col)


                            else:
                                selected_piece = None
                                valid_moves = []
                                movement_moves = []

                        data = pickle.dumps(("MOVE", (row, col)))
                        client_socket.send(data)
                        print("moved")
            else:
                data = client_socket.recv(4096)
                print("recieved")
                event, pos = pickle.loads(data)

                if event == "QUIT":
                    running = False
                elif event == "MOVE":
                    (row, col) = pos
                    if selected_piece:
                        if isinstance(piece, Pawn) and groupchecker((row, col), selected_piece, pawn_group):
                            pawn = chess_board.board[row][col]
                            act_pawn = chess_board.board[selected_piece[0]][selected_piece[1]]
                            pawn_group.append(selected_piece)
                            pawn_group.append((row, col))
                            group_move.append((selected_piece[0] + act_pawn.direction, selected_piece[1]))
                            group_move.append((row + act_pawn.direction, col))
                        else:
                            if (row, col) in group_move:
                                for i in range(len(pawn_group)):
                                    chess_board.move_piece(pawn_group[i], group_move[i])
                                turn_color = PWHITE
                                pawn_group = []
                                group_move = []

                            elif (row, col) in valid_moves:
                                if not isinstance(piece, swordsman):
                                    chess_board.move_piece(selected_piece, (row, col))
                                    turn_color = PWHITE
                                else:
                                    if chess_board.board[row][col] != None:
                                        if chess_board.board[row][col].color != turn_color:
                                            chess_board.board[row][col] = None
                                            turn_color = PWHITE

                            elif (row, col) in movement_moves:
                                chess_board.move_piece(selected_piece, (row, col))
                                turn_color = PWHITE

                            selected_piece = None
                            valid_moves = []
                            pawn_group = []
                            group_move = []
                    else:
                        piece = chess_board.board[row][col]
                        if piece and piece.color == turn_color:
                            selected_piece = (row, col)
                            valid_moves = chess_board.get_valid_moves(piece, row, col)
                            if not isinstance(piece, swordsman):
                                movement_moves = []
                            else:
                                movement_moves = valid_moves
                                valid_moves = chess_board.swordkill(piece, row, col)


                        else:
                            selected_piece = None
                            valid_moves = []
                            movement_moves = []

            screen.fill(BLACK)

            if len(group_move) != 0:
                chess_board.draw(screen, selected_piece, valid_moves, movement_moves, group_move, True)
            else:
                chess_board.draw(screen, selected_piece, valid_moves, movement_moves, group_move, False)

            pygame.display.flip()

        draw_midline(screen)

        pygame.quit()
        client_socket.close()


if __name__ == '__main__':
    run_client()

