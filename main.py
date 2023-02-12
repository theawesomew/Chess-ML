import pickle
import torch
import chess
from train import Model
from state import Board

def minimax (model, position, depth=4, turn=False):
    if depth == 0 or position.is_game_over():
        return [model(torch.from_numpy(Board(position).serialize()).float()).item()]
    else:
        best, bestmove = None, None
        if turn:
            best = -1e6
        else:
            best = 1e6

        for i, move in enumerate(position.legal_moves):
            position.push(move)

            current = minimax(model, position, depth-1, not turn)[0]

            if turn:
                if current > best:
                    best = current
                    bestmove = move
            else:
                if current < best:
                    best = current
                    bestmove = move

            position.pop()

        return [best, bestmove]

if __name__ == '__main__':
    f = open('model.pickle', 'rb')
    m = pickle.load(f)
    b = chess.Board("r1bqkbnr/ppp2ppp/2np4/8/2B1P3/5Q2/PB3PPP/RN2K1NR w KQkq - 0 7")
    print(b)
    print(m(torch.from_numpy(Board(b).serialize()).float()).item())
