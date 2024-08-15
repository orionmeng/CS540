import random
import copy
import time

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def succ(self, state):
        """ Function to score each of the successor states.
        """
        successors = []
        if self.is_drop_phase(state):
            for r in range(5):
                for c in range(5):
                    if state[r][c] == ' ':
                        new_state = [row[:] for row in state]
                        new_state[r][c] = self.my_piece
                        successors.append(new_state)
        else:
            for r in range(5):
                for c in range(5):
                    if state[r][c] == self.my_piece:
                        adjacent_positions = [(r + dr, c + dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (dr, dc) != (0, 0)]
                        for pos in adjacent_positions:
                            if 0 <= pos[0] < 5 and 0 <= pos[1] < 5 and state[pos[0]][pos[1]] == ' ':
                                new_state = [row[:] for row in state]
                                new_state[r][c] = ' '
                                new_state[pos[0]][pos[1]] = self.my_piece
                                successors.append(new_state)
        return successors

    def heuristic_game_value(self, state):
        """ Function to evaluate non-terminal states.
        """
        if self.game_value(state) != 0:
            return self.game_value(state)

        score = 0.0

        for row in state:
            score += row.count(self.my_piece) * 0.1
            score -= row.count(self.opp) * 0.1

        for col in range(5):
            col_values = [state[r][col] for r in range(5)]
            score += col_values.count(self.my_piece) * 0.1
            score -= col_values.count(self.opp) * 0.1

        return score

    def max_value(self, state, depth):
        """ Minimax function for the maximizing player.
        """
        if self.game_value(state) != 0 or depth >= 3:
            return self.heuristic_game_value(state)

        v = float('-inf')
        successors = self.succ(state)
        for successor in successors:
            v = max(v, self.min_value(successor, depth + 1))
        return v

    def min_value(self, state, depth):
        """ Minimax function for the minimizing player.
        """
        if self.game_value(state) != 0 or depth >= 3:
            return self.heuristic_game_value(state)

        v = float('inf')
        successors = self.succ(state)
        for successor in successors:
            v = min(v, self.max_value(successor, depth + 1))
        return v

    def is_drop_phase(self, state):
        piece_count = sum([1 for row in state for cell in row if cell in self.pieces])
        return piece_count < 8

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        drop_phase = self.is_drop_phase(state)

        move = []


        if drop_phase:
            max_score = float('-inf')
            best_move = None

            for r in range(5):
                for c in range(5):
                    if state[r][c] == ' ':
                        new_state = [row[:] for row in state]
                        new_state[r][c] = self.my_piece
                        score = self.heuristic_game_value(new_state)
                        if score > max_score:
                            max_score = score
                            best_move = [(r, c)]

            move = best_move

        else:
            max_score = float('-inf')
            best_move = None
            best_source = None

            for r in range(5):
                for c in range(5):
                    if state[r][c] == self.my_piece:
                        adjacent_positions = [(r + dr, c + dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (dr, dc) != (0, 0)]
                        for pos in adjacent_positions:
                            if 0 <= pos[0] < 5 and 0 <= pos[1] < 5 and state[pos[0]][pos[1]] == ' ':
                                new_state = [row[:] for row in state]
                                new_state[r][c] = ' '
                                new_state[pos[0]][pos[1]] = self.my_piece
                                score = self.heuristic_game_value(new_state)
                                if score > max_score:
                                    max_score = score
                                    best_move = [(pos[0], pos[1])]
                                    best_source = (r, c)

            if best_source:
                move = best_move
                move.append(best_source)

        return move

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # check \ diagonal wins
        for r in range(2):
            for c in range(2):
                if (
                    state[r][c] != ' ' and
                    state[r][c] == state[r+1][c+1] == state[r+2][c+2] == state[r+3][c+3]
                ):
                    return 1 if state[r][c] == self.my_piece else -1

        # check / diagonal wins
        for r in range(2):
            for c in range(3, 5):
                if (
                    state[r][c] != ' ' and
                    state[r][c] == state[r+1][c-1] == state[r+2][c-2] == state[r+3][c-3]
                ):
                    return 1 if state[r][c] == self.my_piece else -1

        # check box wins
        for r in range(4):
            for c in range(4):
                if (
                    state[r][c] != ' ' and
                    state[r][c] == state[r][c+1] == state[r+1][c] == state[r+1][c+1]
                ):
                    return 1 if state[r][c] == self.my_piece else -1

        return 0 # no winner yet

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
