"""Tic Tac Toe Game implementation"""
import re
from typing import Tuple, List, Optional

import numpy as np


class TicTacToeBoard:
    """Tic Tac Toe Board

    Contains the board state at a single point in time, i.e. 9 squares with 3
    possible values [-,x,o].
    """

    board_state: np.ndarray
    """Board state as a numpy array of shape (3,3)

    Note we use integers to represent the state of each square, rather than
    strings, which are defined below."""

    blank: int = 0
    """Integer representation of blank on the board (numpy array)"""

    x: int = 1
    """Integer representation of x on the board (numpy array)"""

    o: int = 2
    """Integer representation of o on the board (numpy array)"""

    def __init__(self, string=None):
        # Initialise an empty board
        self.board_state = np.full((3, 3), self.blank, int)

        # Setup the mapping of strings to square values
        self.map = {self.x: 'x', self.o: 'o', self.blank: '-'}

        # Parse a string representation of the board state, if given
        if string is not None:
            self.parse_str(string)

    def get_valid_moves(self):
        ''' return a list of valid (i,j,player) moves '''
        # work out whose turn it is
        num_x = np.sum(self.board_state == self.x)
        num_o = np.sum(self.board_state == self.o)
        if num_x == num_o:
            turn = self.x
        elif num_x == num_o + 1:
            turn = self.o
        else:
            print("Invalid board state")

        # make list
        l = []
        for i in range(3):
            for j in range(3):
                if self.board_state[i, j] == self.blank:
                    l.append((i, j, turn))
        return l

    def make_move(self, i, j, player):
        # check if legal
        if i >= 3 or i < 0 or j >= 3 or j < 0:
            print("Index out of bounds")
        elif self.board_state[i, j] != self.blank:
            print("Not a blank square")
        elif player != self.x and player != self.o:
            print("Invalid player")

        # modify board
        self.board_state[i, j] = player

    def check_win(self):
        for player in [self.x, self.o]:
            won = False
            # check columns
            if np.any(np.all(self.board_state == player, axis=0)):
                won = True
            # check rows
            if np.any(np.all(self.board_state == player, axis=1)):
                won = True

            # check diagonals
            elif np.all(np.diag(self.board_state) == player) \
                    or np.all(np.diag(np.fliplr(self.board_state)) == player):
                won = True

            if won:
                return player

        return False

    def __str__(self):
        b = self.board_state
        out = ''
        # convert state to string
        for i in range(3):
            for j in range(3):
                out += f" {self.map[b[i,j]]}"
            out += "\n"
        return out

    def get_counts(self):
        blank_count = np.sum(self.board_state == self.blank)
        x_count = np.sum(self.board_state == self.x)
        o_count = np.sum(self.board_state == self.o)
        return blank_count, x_count, o_count

    def validate_state(self):
        """ Check that the board state is valid, including that the number of
        x's and o's is correct.
        """
        blank_count, x_count, o_count = self.get_counts()
        assert blank_count + x_count + o_count == 9
        assert (x_count == o_count) or (x_count == o_count + 1)

        return True

    def validate_str(self, string):
        """ Check that the string representation of the board state is valid. """
        lines = string.strip("\n").split("\n")

        # Check there are 3 lines
        if len(lines) != 3:
            raise ValueError("Invalid game string - incorrect number of rows")

        # Check that each line is in the correct format
        for line in lines:
            if re.match(r"^[ ]*[-xo][ ][-xo][ ][-xo][ ]*$", line) is None:
                raise ValueError("Invalid game string - invalid row format")

        return True

    def parse_str(self, string: str):
        # Ensure the state string is of the correct format
        self.validate_str(string)

        # Split the string into lines
        lines = string.strip("\n").split("\n")

        # create a dict that does the opposite of self.map
        rev_map = {v: k for k, v in self.map.items()}

        # iterate over it and convert to state
        for i, line in enumerate(lines):
            l = line.strip(" ").split(" ")
            for j, char in enumerate(l):
                self.board_state[i, j] = rev_map[char]


class TicTacToeGame:
    """ Class to represent a game of Tic Tac Toe at multiple points in time.
    """

    def __init__(self,
                 init_board: Optional[str] = None,
                 game_string: Optional[str] = None,
                 check_valid_move: bool = True,
                 check_valid_state: bool = True):

        self.board = TicTacToeBoard(init_board)
        self.history: List[TicTacToeBoard] = [self.board]

        # Choose which validity checks to perform
        #  Check that the board state has only 'x', 'o', and '-' characters
        self.check_valid_string = True

        #  Check that on state change, only one blank piece has changed
        self.check_valid_move = check_valid_move

        # Check that the board state is valid, such that #x == #o or #x == #o +
        # 1
        self.check_valid_state = check_valid_state

    def reset(self):
        self.board = TicTacToeBoard()
        self.history = [self.board]
        return self

    def validate_move(self, old_state: TicTacToeBoard,
                      new_state: TicTacToeBoard):
        if self.check_valid_state:
            assert old_state.validate_state()
            assert new_state.validate_state()

        if self.check_valid_move:
            # Check that the new state has one more piece than the old state
            assert np.sum(new_state.board_state != old_state.board_state) == 1
            assert np.sum(new_state.board_state == old_state.board_state) == 8

            old_blank_count, _old_x_count, _old_o_count = old_state.get_counts()
            new_blank_count, _new_x_count, _new_o_count = new_state.get_counts()
            assert new_blank_count == old_blank_count - 1

    def add_state(self, board_string: str) -> Tuple[int, bool]:
        """ Add a new state to the game, check for validity.

        Returns:
            outcome (int): The outcome of the game, 0 if the game is not over
            valid (bool): Whether the state was valid
        """

        # Perform validity checks
        try:
            # Load the new state and save it to the history
            self.board = TicTacToeBoard(board_string)
            self.history.append(self.board)

            if self.check_valid_state:
                assert self.board.validate_state()

            if self.check_valid_move and len(self.history) > 1:
                self.validate_move(self.history[-2], self.board)

        except ValueError as _err:
            return 0, False

        except AssertionError as _err:
            return 0, False

        # If valid, perform win checks
        outcome = self.board.check_win()
        return outcome, True

    def validate_game_string(
            self, game_string: str) -> Tuple[bool, float, int]:
        """Validate the game string

        Args:
            game_string: Game string

        Returns:
            Tuple containing isValid, the final outcome and the turn.
        """
        self.board = TicTacToeBoard()
        self.history = [self.board]

        # split game string into board states
        board_strings = game_string.split("\n\n")[1:]

        # Make sure there are at least 2 board states
        if len(board_strings) == 0:
            print('num boards:', len(board_strings))
            return False, 0, 0

        turn: int = 0
        final_outcome: float = 0
        final_turn: int = 0

        for board_string in board_strings:
            outcome, valid = self.add_state(board_string)
            turn += 1

            # If the game is won, return score based on winner
            if outcome and not final_outcome:
                final_outcome = outcome
                final_turn = turn

            if not valid:
                return False, 0, turn

        turn = turn if (final_turn == 0) else final_turn

        return True, final_outcome, turn

    def evaluate_game_string(self, game_string: str) -> float:
        valid, outcome, turn = self.validate_game_string(game_string)

        # If the game is not valid, return -1
        if not valid:
            return 0.0

        if not outcome:
            return 0.5

        #  Extra rules to better differentiate possible wins
        reward_delta = 0.05 * (9 - turn)
        if turn <= 5:
            reward_delta = 0.2

        # If the game is won, return score based on winner
        if outcome == self.board.o:
            return 0.8 + reward_delta  #  [ 0.8, 1.0]
        if outcome == self.board.x:
            return 0.2 - reward_delta  #  [ 0.0, 0.2]

        return 0.0

    def extract_game_string(self, game_string: str) -> str:
        """ Extract the game string from a full game string

        Args:
            game_string (str): Full game string

        Returns:
            str: Game string
        """
        game_string_after_start = game_string.split(
            "Let's play Tic Tac Toe:\n")[1]
        game_string_before_end = game_string_after_start.split("\n<|endoftext|>")[
            0]
        return game_string_before_end


def generate_random_game():
    b = TicTacToeBoard()
    game_state_history = [str(b)]
    for t in range(9):
        valid_moves = b.get_valid_moves()
        move = np.random.choice(len(valid_moves))
        b.make_move(*valid_moves[move])
        game_state_history.append(str(b))

    return "Let's play Tic Tac Toe:\n" + \
        "\n".join(game_state_history) + "<|endoftext|>"


def generate_dataset(number_games: int) -> List[str]:
    """Generate a list of games

    Args:
        number_games (int): Number of games

    Returns:
        List: List of games (strings with a full game)
    """
    return [generate_random_game() for _ in range(number_games)]


if __name__ == "__main__":
    # TEST
    # Generate a game
    game_string = generate_random_game()
    print(game_string)

    # Evaluate the game
    g = TicTacToeGame()
    game_string = g.extract_game_string(game_string)
    print("%.2f" % g.evaluate_game_string(game_string))
