import numpy as np
import chess
import chess.pgn
import os
from sys import argv
from state import Board

def parse_result (x):
    return float({"1-0":1,"1/2-1/2":0,"0-1":-1}[x])

max_games = int(argv[1])

for fn in os.listdir("data"):
    with open(os.path.join("data",fn)) as f:
        X = []
        total = 0
        while (game := chess.pgn.read_game(f)) != None and total < max_games:
            total += 1
            b = chess.Board()
            result = parse_result(game.headers["Result"])
            for move in game.mainline_moves():
                b.push(move)
                value = Board(b).serialize().astype(np.float32).tolist()
                value.append(result)
                X.append(value)
                

X = np.array(X, dtype=np.float64)                
print(X)
np.savetxt('dataset.csv', X, delimiter=',')
