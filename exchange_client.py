import asyncio
import datetime
import aiohttp
import ccxt.async_support as ccxt
from ccxt import NetworkError, RateLimitExceeded, ExchangeNotAvailable
import logging
import pandas as pd
import requests

class ExchangeClient:
    def __init__(self, exchange_id, api_key, api_secret):
        params = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        }

        self.exchange = getattr(ccxt, exchange_id)(params)
        self.name = exchange_id
        logging.info(f"Initialized exchange client for {exchange_id}")

    async def fetch_with_retry(self, method, args=(), retries=5, initial_delay=1.0):
        delay = initial_delay
        for attempt in range(retries):
            try:
                return await getattr(self.exchange, method)(*args)
            except (RateLimitExceeded, ExchangeNotAvailable, NetworkError) as e:
                if attempt < retries - 1:
                    logging.warning(f"Attempt {attempt + 1} failed with error: {e}. Retrying in {delay} seconds.")
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    logging.error(f"Final attempt failed with error: {e}. No more retries.")
                    raise
            except Exception as e:
                logging.error(f"Unexpected error during API call: {e}")
                raise
        raise Exception("Max retries exceeded for API call")


    async def get_price_history_length(self, symbol, start_str):
        symbol = symbol.replace('/', '')
        url = 'https://api.binance.com/api/v3/klines'
        params = {
            'symbol': symbol,
            'interval': '1d',
            'limit': 31,
            'startTime': int(start_str.timestamp() * 1000)
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                # Ensure the response is successful
                if response.status != 200:
                    data = await response.text()
                    print(f"Error fetching data for {symbol}: {data}")
                    return 0

                data = await response.json()

                if not data:
                    print(f"No data returned for {symbol}")
                    return 0

                earliest_date = datetime.datetime.fromtimestamp(data[0][0] / 1000)

                today = datetime.datetime.now()
                history_length_days = (today - earliest_date).days

                return history_length_days

    async def get_binance_klines(self, symbol, interval, start_str, end_str=None):
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': int(start_str.timestamp() * 1000),
        }
        if end_str:
            params['endTime'] = int(end_str.timestamp() * 1000)

        response = requests.get(url, params=params)
        data = response.json()
        df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df['volume'] = pd.to_numeric(df['volume'])
        df['quote_asset_volume'] = pd.to_numeric(df['quote_asset_volume'])
        return df

    async def fetch_historical_data(self, symbol, start_date=None, end_date=None, timeframe='1m', limit=1000):
        try:
            since = int(start_date.timestamp() * 1000) if start_date else None
            ohlcv = await self.fetch_with_retry('fetch_ohlcv', (symbol, timeframe, since, limit))
            logging.info(f"Fetched historical data for {symbol} timeframe={timeframe} points={len(ohlcv)} since={since}")
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

            filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            return filtered_df
        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol}: {e}")
            raise

    async def fetch_historical_data_chunked(client, symbol, start_date, end_date, timeframe='1m', limit=1000):
        historical_data = []
        since = int(start_date.timestamp() * 1000)
        while True:
            ohlcv = await client.fetch_with_retry('fetch_ohlcv', (symbol, timeframe, since, limit))
            if not ohlcv:
                break

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            historical_data.append(df)

            since = df['timestamp'].iloc[-1] + 1  # Add 1 millisecond to move to the next chunk

            if df['date'].iloc[-1] > end_date:
                break

        if not historical_data:
            empty_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'date'])
            logging.info(f"chunked return empty df for {symbol} with schema {list(empty_df.columns)}")
            return empty_df
        full_data = pd.concat(historical_data, ignore_index=True)

        filtered_data = full_data[(full_data['date'] >= start_date) & (full_data['date'] <= end_date)]
        logging.info(
            f"chunked filtered: {symbol} tf={timeframe} start={start_date} end={end_date} shape={filtered_data.shape}, first={filtered_data['date'].min() if not filtered_data.empty else None}, last={filtered_data['date'].max() if not filtered_data.empty else None}"
        )

        return filtered_data

    async def close(self):
        await self.exchange.close()
        logging.info(f"Closed exchange client for {self.exchange.id}")

