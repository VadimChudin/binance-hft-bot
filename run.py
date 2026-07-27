"""Entry point for the Binance USD-M futures scalping bot.

Usage:
    python run.py                 # use config.yaml
    python run.py --config my.yaml
    python run.py --mode paper    # override mode
"""

from __future__ import annotations

import argparse
import asyncio
import signal

from hftbot.config import load_config
from hftbot.engine import TradingEngine
from hftbot.logger import get_logger, setup_logging

log = get_logger("run")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binance USD-M futures scalping bot")
    p.add_argument("--config", default="config.yaml", help="path to YAML config")
    p.add_argument("--mode", choices=["paper", "live"], help="override execution mode")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode:
        config.mode = args.mode

    setup_logging(config.logging.level, config.logging.dir)
    engine = TradingEngine(config)

    loop = asyncio.get_running_loop()
    stop_requested = asyncio.Event()

    def _request_stop() -> None:
        log.info("stop signal received")
        stop_requested.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            # Windows: signal handlers on the loop are limited; rely on KeyboardInterrupt.
            pass

    engine_task = asyncio.create_task(engine.run())
    stop_task = asyncio.create_task(stop_requested.wait())
    done, _ = await asyncio.wait(
        {engine_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
    )
    if stop_task in done:
        await engine.stop()
        await engine_task
    else:
        stop_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
