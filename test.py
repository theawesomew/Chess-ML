import asyncio
import chess
import chess.engine

ENGINE_PATH = "engine/stockfish_15.1_win_x64_avx2/stockfish-windows-2022-x86-64-avx2.exe"

async def evaluate (position, engine):
    return (await engine.analyse(position, chess.engine.Limit(depth=20)))["score"].relative.score(mate_score=10000) * 0.0001

async def main () -> None:
    transport, engine = await chess.engine.popen_uci(ENGINE_PATH)
    evaluation = await evaluate(chess.Board(), engine)
    await engine.quit()
       

if __name__ == "__main__":
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(main())
