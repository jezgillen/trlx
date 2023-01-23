import numpy as np
import pytest

from soft_optim.game import TicTacToeBoard, TicTacToeGame


class TestTicTacToeBoardValidStates:
    """Test valid states for TicTacToeBoard"""

    mock_empty_board_str: str = \
        """- - -
        - - -
        - - -"""

    mock_x_won_board_str: str = \
        """x x x
        o - o
        - o -"""

    def test_initializes_empty_board(self):
        board = TicTacToeBoard()
        expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(
            board.board_state,
            expected)
        assert board.check_win() == False
        assert board.validate_state()

    def test_parses_empty_board(self):
        board = TicTacToeBoard(self.mock_empty_board_str)
        expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(
            board.board_state,
            expected)
        assert board.check_win() == False
        assert board.validate_state()

    def test_parses_x_won_board(self):
        board = TicTacToeBoard(self.mock_x_won_board_str)
        expected = np.array([[1, 1, 1], [2, 0, 2], [0, 2, 0]])
        np.testing.assert_array_equal(
            board.board_state,
            expected)
        assert board.check_win() == board.x
        assert board.validate_state()

class TestTicTacToeBoardInvalidStates:
    """Test invalid states for TicTacToeBoard"""

    def test_parser_errors_too_many_lines(self):
        with pytest.raises(ValueError, match='Invalid'):
            invalid_str: str = \
                """x x x
                o - o
                - o -
                x x x"""
            TicTacToeBoard(invalid_str)

    def test_parser_errors_too_many_columns(self):
        with pytest.raises(ValueError, match='Invalid'):
            invalid_str: str = \
                """x x x x
                o - o
                - o -"""
            TicTacToeBoard(invalid_str)

    def test_parser_errors_invalid_row_character(self):
        with pytest.raises(ValueError, match='Invalid'):
            invalid_str: str = \
                """x x y
                o - o
                - o -"""
            TicTacToeBoard(invalid_str)

    def test_parser_errors_x_won_cheat_board(self):
        mock_x_won_cheat_board_str: str = \
        """x x x
        - o -
        - - -"""
        board = TicTacToeBoard(mock_x_won_cheat_board_str)
        expected = np.array([[1, 1, 1], [0, 2, 0], [0, 0, 0]])
        np.testing.assert_array_equal(
            board.board_state,
            expected)
        assert board.check_win() == board.x
        with pytest.raises(AssertionError):
            board.validate_state()

class TestTicTacToeGameValidGames:
    """Test invalid games for TicTacToeGame"""
    mock_game_x_win_str: str = \
        """- - -
        - - -
        - - -

        - - -
        - x -
        - - -

        - - -
        - x o
        - - -

        - - -
        - x o
        - x -

        - - -
        - x o
        - x o

        - x -
        - x o
        - x o"""

    def test_parses_valid_game(self):
        game = TicTacToeGame(check_valid_move=True, check_valid_state=True)
        valid, outcome, turn = game.validate_game_string(self.mock_game_x_win_str)
        assert outcome == game.board.x
        assert turn == 5
        assert valid

class TestTicTacToeGameInvalidGames:
    mock_game_o_win_invalid_wrong_player_place_many: str = \
        """- - -
        - - -
        - - -

        - - -
        o o o
        - - -"""

    mock_game_x_win_invalid_place_many: str = \
        """- - -
        - - -
        - - -

        - - -
        - x -
        - - -

        - - -
        - x o
        - x -

        - x x
        - x o
        - x -"""

    mock_game_x_win_invalid_wrong_player: str = \
        """- - -
        - - -
        - - -

        - - -
        - x -
        - - -

        - x -
        - x -
        - - -

        - x -
        - x -
        - x -"""

    mock_garbage_string: str = \
        """- 's's's's's's's's's's's's's's's's's's's's's's's's's's's's's's"""

    def test_game_invalid_wrong_player_place_many(self):
        game = TicTacToeGame(check_valid_move=False, check_valid_state=False)
        valid, outcome, turn = game.validate_game_string(self.mock_game_o_win_invalid_wrong_player_place_many)
        assert valid and game.board.o == outcome
        assert turn == 1

        game = TicTacToeGame(check_valid_move=True, check_valid_state=False)
        valid, outcome, turn = game.validate_game_string(self.mock_game_o_win_invalid_wrong_player_place_many)
        assert not valid

        game = TicTacToeGame(check_valid_move=False, check_valid_state=True)
        valid, outcome, turn = game.validate_game_string(self.mock_game_o_win_invalid_wrong_player_place_many)
        assert not valid

        game = TicTacToeGame(check_valid_move=True, check_valid_state=True)
        valid, outcome, turn = game.validate_game_string(self.mock_game_o_win_invalid_wrong_player_place_many)
        assert not valid

    def test_game_invalid_wrong_player(self):
        game = TicTacToeGame(check_valid_move=False, check_valid_state=False)
        valid, outcome, turn = game.validate_game_string(self.mock_game_x_win_invalid_wrong_player)
        assert valid
        assert game.board.x == outcome
        assert turn == 3

        game = TicTacToeGame(check_valid_move=True, check_valid_state=False)
        valid, outcome, turn = game.validate_game_string(self.mock_game_x_win_invalid_wrong_player)
        assert valid
        assert game.board.x == outcome
        assert turn == 3

        game = TicTacToeGame(check_valid_move=False, check_valid_state=True)
        valid, outcome, turn = game.validate_game_string(self.mock_game_x_win_invalid_wrong_player)
        assert not valid

        game = TicTacToeGame(check_valid_move=True, check_valid_state=True)
        valid, outcome, turn = game.validate_game_string(self.mock_game_x_win_invalid_wrong_player)
        assert not valid

    def test_game_invalid_place_many(self):
        game = TicTacToeGame(check_valid_move=False, check_valid_state=False)
        valid, outcome, turn = game.validate_game_string(self.mock_game_x_win_invalid_place_many)
        assert valid
        assert game.board.x == outcome
        assert turn == 3

        game = TicTacToeGame(check_valid_move=True, check_valid_state=False)
        valid, outcome, turn = game.validate_game_string(self.mock_game_x_win_invalid_place_many)
        assert not valid

        # should return true??
        game = TicTacToeGame(check_valid_move=False, check_valid_state=True)
        valid, outcome, turn = game.validate_game_string(self.mock_game_x_win_invalid_place_many)
        assert not valid

        game = TicTacToeGame(check_valid_move=True, check_valid_state=True)
        valid, outcome, turn = game.validate_game_string(self.mock_game_x_win_invalid_place_many)
        assert not valid

    def test_garbage_string(self):
        game = TicTacToeGame(check_valid_move=False, check_valid_state=False)
        valid, outcome, turn = game.validate_game_string(self.mock_garbage_string)
        assert not valid

        game = TicTacToeGame(check_valid_move=True, check_valid_state=True)
        valid, outcome, turn = game.validate_game_string(self.mock_garbage_string)
        assert not valid