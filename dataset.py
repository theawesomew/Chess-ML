import asyncio
import numpy as np
import chess
import chess.pgn
import chess.engine
import os
import math
from tqdm import tqdm
from sys import argv
from state import Board

def parse_result (x):
    return float({"1-0":1,"1/2-1/2":0,"0-1":-1}[x])

def normalize (x):
    return 2/math.pi * math.atan(x)

max_games = int(argv[1])
ENGINE_PATH = "engine/stockfish_15.1_win_x64_avx2/stockfish-windows-2022-x86-64-avx2.exe"

async def evaluate (board, engine):
    analysis = await engine.analyse(board, chess.engine.Limit(depth=20))
    value = analysis["score"].relative.score(mate_score=10000)/100
    return normalize(value)    

async def main() -> None:
    transport, engine = await chess.engine.popen_uci(ENGINE_PATH)

    for fn in os.listdir("data"):
        with open(os.path.join("data",fn)) as f:
            X = []
            total = 0
            pbar = tqdm(total=max_games)
            while (game := chess.pgn.read_game(f)) != None and total < max_games:
                total += 1
                b = chess.Board()
                for move in game.mainline_moves():
                    b.push(move)
                    result = await evaluate(b, engine)
                    value = Board(b).serialize().astype(np.float32).tolist()
                    value.append(result)
                    X.append(value)
                pbar.update(1)
            
            pbar.close()
                

                    
    await engine.quit()

    X = np.array(X, dtype=np.float64)                
    print(X)
    np.savetxt('dataset.csv', X, delimiter=',')

if __name__ == "__main__":
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy()) 
    asyncio.run(main())
