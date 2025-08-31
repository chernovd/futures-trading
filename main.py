import asyncio
import datetime
from exchange_client import ExchangeClient
from trading import TradingBot
import config
import logging
import traceback

async def main():
    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting trading bot...")

    try:
        client = ExchangeClient('binance', config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
        trading_bot = TradingBot(client, config.TRADING_PAIRS)

        start_date = datetime.datetime.fromisoformat(config.BACKTEST_START)
        end_date = datetime.datetime.fromisoformat(config.BACKTEST_END)

        await trading_bot.run_backtest(start_date, end_date)
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main function: {e}, Traceback: {traceback.format_exc()}")

    finally:
        await client.close()
        logging.info("Trading bot shut down.")

if __name__ == "__main__":
    asyncio.run(main())