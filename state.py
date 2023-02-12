import numpy as np
import chess

class Board ():

    def __init__ (self, board):
        self.board = board

    def serialize (self):
        # Squares (2 * 7 + 1 = 15) (4 bits each)
        # - EMPTY 0000
        # - PAWN (0001 | 1001)
        # - BISHOP (0010 | 1010)
        # - KNIGHT (0011 | 1011)
        # - ROOK (ABLE TO CASTLE) (0100 | 1100)
        # - ROOK (UNABLE TO CASTLE) (0101 | 1101)
        # - QUEEN (0110 | 1110)
        # - KING (0111 | 1111)
        # Extra state
        # - En passant (1 bit)
        # - Move (1 bit)

        assert self.board.is_valid()

        bitboard = np.zeros(64, dtype=np.uint8)

        for i in range(64):
            if (piece := self.board.piece_at(i)) is not None:
                bitboard[i] = {"P": 1, "B": 2, "N": 3, "R": 4, "Q": 6, "K": 7, \
                               "p": 8, "b": 9, "n": 10, "r": 11, "q": 13, "k": 14 }[piece.symbol()]
        
        if self.board.has_kingside_castling_rights(chess.BLACK):
            assert bitboard[63] == 11
            bitboard[63] = 12
        if self.board.has_queenside_castling_rights(chess.BLACK):
            assert bitboard[56] == 11
            bitboard[56] = 12
        if self.board.has_kingside_castling_rights(chess.WHITE):
            assert bitboard[7] == 4
            bitboard[7] = 5
        if self.board.has_queenside_castling_rights(chess.WHITE):
            assert bitboard[0] == 4
            bitboard[0] = 5

        state = np.zeros((4, 8, 8), dtype=np.uint8)

        bitboard = bitboard.reshape((8,8))

        for j in range(4):
            state[j] = (bitboard >> (3-j)) & 1

        state = np.append(state.reshape(-1), [self.board.turn == chess.WHITE])

        return state
        

