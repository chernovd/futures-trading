import asyncio
import json
import logging
import os
import traceback
import aiohttp

import requests
import config
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import talib
from typing import List, Dict, Any
import requests

class TradingBot:

    def __init__(self, client, trading_pairs):
        self.client = client
        self.trading_pairs = trading_pairs

        self.lock = asyncio.Lock()

        self.usdt_balance = 0.0
        self.total_net_profit = 0.0
        self.total_loss = 0.0

        self.trend_states = { 'supports': None, 'resistance': None}
        self.previous_indicators = {pair: {'rsi_history': []} for pair in trading_pairs}

        self.visualization_data_1h = {pair: pd.DataFrame() for pair in trading_pairs}
        self.visualization_data_1m = {pair: {'candlestickSeries': [], 'supportLevels': [], 'resistanceLevels': [], 'tradeMarkers': []} for pair in trading_pairs}
        self.any_trade_open = False
        self.signals = { pair: { 'overbought': False, 'oversold': False} for pair in trading_pairs}
        self.positions = {}
        self.trades = {pair: [] for pair in self.trading_pairs}
        self.open_trades = {}

        logging.info("TradingBot initialized.")

    def update_trend_state(self, pair, support_levels, resistance_levels, current_date_1m):
            # Initialize the trend state for the pair if it doesn't exist
            if pair not in self.trend_states:
                self.trend_states[pair] = {
                    'supports': [],
                    'resistances': [],
                    'complete_trend': False
                }

            state = self.trend_states[pair]
            print('state:', state)
            if support_levels:
                last_support = support_levels[-1]
                last_support_time = datetime.datetime.strptime(last_support[0], '%Y-%m-%d %H:%M:%S')
                if len(state['supports']) > 0:
                    penultimate_time = datetime.datetime.strptime(state['supports'][-1][0], '%Y-%m-%d %H:%M:%S')
                if not state['supports'] or (last_support[1] > state['supports'][-1][1] and last_support_time > penultimate_time):
                    if current_date_1m.to_pydatetime() >= last_support_time:
                        state['supports'].append((last_support[0], last_support[1]))

            if resistance_levels:
                last_resistance = resistance_levels[-1]
                last_resistance_time = datetime.datetime.strptime(last_resistance[0], '%Y-%m-%d %H:%M:%S')
                if len(state['resistances']) > 0:
                    penultimate_time = datetime.datetime.strptime(state['resistances'][-1][0], '%Y-%m-%d %H:%M:%S')
                if (state['supports'] and not state['resistances'] and last_resistance_time > datetime.datetime.strptime(state['supports'][-1][0], '%Y-%m-%d %H:%M:%S')) or (state['resistances'] and last_resistance[1] > state['resistances'][-1][1] and last_resistance_time > penultimate_time):
                    if current_date_1m.to_pydatetime() >= last_resistance_time:
                        state['resistances'].append((last_resistance[0], last_resistance[1]))

            state['supports'] = state['supports'][-2:]
            state['resistances'] = state['resistances'][-2:]

            if len(state['supports']) == 2 and len(state['resistances']) == 2:
                state['complete_trend'] = True
            else:
                state['complete_trend'] = False

    def should_long(self, indicators_1m, indicators_1h, price):
        current_indicators_1m = indicators_1m['current']
        previous_indicators_1m = indicators_1m['previous']

        if not self.indicators_ready(current_indicators_1m, previous_indicators_1m):
            return False

        # Filters: trend crossover, RSI midline, MACD above signal, price above mid band, ADX
        # Optional Donchian/ATR breakout strategy
        if getattr(config, 'USE_BREAKOUT_STRATEGY', False):
            # Long breakout: close breaks above Donchian upper and momentum ok
            don_up_prev = indicators_1m['previous'].get('donchian_upper', np.nan)
            don_up_now = current_indicators_1m.get('donchian_upper', np.nan)
            atr_now = current_indicators_1m.get('atr', np.nan)
            # Require volatility floor
            atr_pct_ok = (not np.isnan(atr_now) and atr_now / max(1e-9, current_indicators_1m['close']) >= float(getattr(config, 'MIN_ATR_PCT', 0.0)))
            # Require RSI increasing
            rsi_up = current_indicators_1m['rsi'] > indicators_1m['previous']['rsi']
            # Require uptrend (close above SMA200) and a sufficiently wide Donchian channel
            min_width = float(getattr(config, 'DONCHIAN_WIDTH_MIN_PCT', 0.01))
            don_width_ok = (not np.isnan(don_up_now) and not np.isnan(current_indicators_1m.get('donchian_lower', np.nan)) and
                            (don_up_now - current_indicators_1m['donchian_lower']) / max(1e-9, current_indicators_1m['close']) >= min_width)
            # Trend: prefer SMA200 if available, else SMA50 (sma_long)
            sma_trend = current_indicators_1m.get('sma200')
            if sma_trend is None or np.isnan(sma_trend):
                sma_trend = current_indicators_1m.get('sma_long')
            uptrend_ok = (not np.isnan(sma_trend) and current_indicators_1m['close'] > sma_trend)
            # Apply ATR buffer above Donchian breakout
            atr_buf = 0.0 if atr_now is None or np.isnan(atr_now) else atr_now * float(getattr(config, 'BREAKOUT_ATR_BUFFER_MULT', 0.0))
            breakout_ok = (not np.isnan(don_up_prev) and not np.isnan(don_up_now) and atr_pct_ok and rsi_up and uptrend_ok and don_width_ok and
                           current_indicators_1m['close'] > don_up_prev + atr_buf and
                           current_indicators_1m['macd'][0] > current_indicators_1m['macd'][1] and
                           current_indicators_1m['adx'] >= getattr(config, 'ADX_MIN', 20))
            # Keep existing filters but prioritize breakout
            vol_ok = True
            try:
                vol_ok = (current_indicators_1m['volume'] >= current_indicators_1m['volume_ma'] * getattr(config, 'VOL_MA_MULTIPLIER', 1.0))
            except Exception:
                vol_ok = True
            return breakout_ok and vol_ok

        vol_ok = True
        try:
            vol_ok = (current_indicators_1m['volume'] >= current_indicators_1m['volume_ma'] * getattr(config, 'VOL_MA_MULTIPLIER', 1.0))
        except Exception:
            vol_ok = True

        passed = ( current_indicators_1m['sma_short'] > current_indicators_1m['sma_long'] and
            previous_indicators_1m['sma_short'] <= previous_indicators_1m['sma_long'] and
            current_indicators_1m['rsi'] > 50 and
            current_indicators_1m['macd'][0] > current_indicators_1m['macd'][1] and
            price > current_indicators_1m['middle_band'] and
            current_indicators_1m['adx'] >= getattr(config, 'ADX_MIN', 20) and
            vol_ok )

        # MACD histogram slope filter
        hist = self.previous_indicators.get(('hist_prev_long', indicators_1m['pair']), None)
        current_hist = indicators_1m['current']['histogram']
        slope_ok = True if hist is None else (current_hist - hist) >= getattr(config, 'MACD_HIST_SLOPE_MIN', 0.0)
        self.previous_indicators[('hist_prev_long', indicators_1m['pair'])] = current_hist

        return passed and slope_ok

    def should_short(self, indicators_1m, indicators_1h, price):

        current_indicators_1m = indicators_1m['current']
        previous_indicators_1m = indicators_1m['previous']

        if not self.indicators_ready(current_indicators_1m, previous_indicators_1m):
            return False

        if getattr(config, 'USE_BREAKOUT_STRATEGY', False):
            # Short breakout: close breaks below Donchian lower and momentum ok
            don_lo_prev = indicators_1m['previous'].get('donchian_lower', np.nan)
            don_lo_now = current_indicators_1m.get('donchian_lower', np.nan)
            atr_now = current_indicators_1m.get('atr', np.nan)
            atr_pct_ok = (not np.isnan(atr_now) and atr_now / max(1e-9, current_indicators_1m['close']) >= float(getattr(config, 'MIN_ATR_PCT', 0.0)))
            rsi_down = current_indicators_1m['rsi'] < indicators_1m['previous']['rsi']
            min_width = float(getattr(config, 'DONCHIAN_WIDTH_MIN_PCT', 0.01))
            don_width_ok = (not np.isnan(don_lo_now) and not np.isnan(current_indicators_1m.get('donchian_upper', np.nan)) and
                            (current_indicators_1m['donchian_upper'] - don_lo_now) / max(1e-9, current_indicators_1m['close']) >= min_width)
            sma_trend = current_indicators_1m.get('sma200')
            if sma_trend is None or np.isnan(sma_trend):
                sma_trend = current_indicators_1m.get('sma_long')
            downtrend_ok = (not np.isnan(sma_trend) and current_indicators_1m['close'] < sma_trend)
            atr_buf = 0.0 if atr_now is None or np.isnan(atr_now) else atr_now * float(getattr(config, 'BREAKOUT_ATR_BUFFER_MULT', 0.0))
            breakout_ok = (not np.isnan(don_lo_prev) and not np.isnan(don_lo_now) and atr_pct_ok and rsi_down and downtrend_ok and don_width_ok and
                           current_indicators_1m['close'] < don_lo_prev - atr_buf and
                           current_indicators_1m['macd'][0] < current_indicators_1m['macd'][1] and
                           current_indicators_1m['adx'] >= getattr(config, 'ADX_MIN', 20))
            vol_ok = True
            try:
                vol_ok = (current_indicators_1m['volume'] >= current_indicators_1m['volume_ma'] * getattr(config, 'VOL_MA_MULTIPLIER', 1.0))
            except Exception:
                vol_ok = True
            return breakout_ok and vol_ok

        vol_ok = True
        try:
            vol_ok = (current_indicators_1m['volume'] >= current_indicators_1m['volume_ma'] * getattr(config, 'VOL_MA_MULTIPLIER', 1.0))
        except Exception:
            vol_ok = True

        passed = ( current_indicators_1m['sma_short'] < current_indicators_1m['sma_long'] and
            previous_indicators_1m['sma_short'] >= previous_indicators_1m['sma_long'] and
            current_indicators_1m['rsi'] < 50 and
            current_indicators_1m['macd'][0] < current_indicators_1m['macd'][1] and
            price < current_indicators_1m['middle_band'] and
            current_indicators_1m['adx'] >= getattr(config, 'ADX_MIN', 20) and
            vol_ok )

        hist = self.previous_indicators.get(('hist_prev_short', indicators_1m['pair']), None)
        current_hist = indicators_1m['current']['histogram']
        slope_ok = True if hist is None else (hist - current_hist) >= getattr(config, 'MACD_HIST_SLOPE_MIN', 0.0)
        self.previous_indicators[('hist_prev_short', indicators_1m['pair'])] = current_hist

        return passed and slope_ok

    def indicators_ready(self, indicators, previous_indicators):
        necessary_indicators = {'macd', 'rsi', 'sma_short', 'sma_long'}
        return necessary_indicators.issubset(indicators) and necessary_indicators.issubset(previous_indicators)

    def macd_crossed_above_signal(self, indicators, previous_indicators):
        return previous_indicators['macd'][0] < previous_indicators['macd'][1] and indicators['macd'][0] >= indicators['macd'][1]

    def macd_crossed_below_signal(self, indicators, previous_indicators):
        return previous_indicators['macd'][0] > previous_indicators['macd'][1] and indicators['macd'][0] <= indicators['macd'][1]

    def macd_rising(self, indicators, previous_indicators):
        return indicators['macd'][0] > previous_indicators['macd'][0]

    def macd_slope(self, indicators, previous_indicators):
        macd_slope = indicators['macd'][0] - previous_indicators['macd'][0]
        signal_slope = indicators['macd'][1] - previous_indicators['macd'][1]

        return macd_slope > 0 and macd_slope > signal_slope and indicators['macd'][0] < indicators['macd'][1]

    async def get_usdt_balance(self):
        return await self.client.get_balance('USDT')

    async def fetch_historical_data(self, start_date, end_date, timeframe=config.TIMEFRAME):
        tasks = [self.client.fetch_historical_data_chunked(pair, start_date, end_date, timeframe) for pair in self.trading_pairs]

        results = await asyncio.gather(*tasks)
        historical_data = {pair: pd.DataFrame(data) for pair, data in zip(self.trading_pairs, results)}

        return historical_data

    async def fetch_market_data(self):
        market_data = await self.client.fetch_tickers()
        return market_data

    async def select_top_altcoins_for_backtesting(self, start_date, end_date, top_n=5) -> List[str]:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.binance.com/api/v3/exchangeInfo") as response:
                exchange_info = await response.json()
        symbols = [
            s['symbol'] for s in exchange_info['symbols']
            if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT'
            and s['symbol'] not in config.EXCLUDE_TRADING_PAIRS
        ]

        kline_tasks = {symbol: self.client.get_binance_klines(symbol, '1d', start_date, end_date) for symbol in symbols}
        kline_data = {}
        for symbol, task in kline_tasks.items():
            try:
                df = await task
                if not df.empty:
                    kline_data[symbol] = df
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")

        volumes = {symbol: df['quote_asset_volume'].sum() for symbol, df in kline_data.items()}
        sorted_symbols = sorted(volumes.items(), key=lambda x: x[1], reverse=True)[:top_n]

        top_symbols = [symbol for symbol, _ in sorted_symbols]

        stable_or_increasing_altcoins = [
            symbol for symbol in top_symbols
            if float(kline_data[symbol].iloc[-1]['close']) >= float(kline_data[symbol].iloc[-1]['open']) * 1.01
        ]

        self.trading_pairs = [f"{symbol[:-4]}/{symbol[-4:]}" for symbol in stable_or_increasing_altcoins]

        for pair in self.open_trades.keys():
            if self.open_trades[pair] is not None and pair not in self.trading_pairs:
                self.trading_pairs.append(pair)
        self.open_trades = {pair: self.open_trades.get(pair, None) for pair in self.trading_pairs}
        self.visualization_data_1m = {pair: {'candlestickSeries': [], 'supportLevels': [], 'resistanceLevels': [], 'tradeMarkers': []} for pair in self.trading_pairs}
        self.visualization_data_1h = {pair: pd.DataFrame() for pair in self.trading_pairs}
        self.signals = { pair: { 'overbought': False, 'oversold': False} for pair in self.trading_pairs}
        self.positions = {pair: self.positions[pair] for pair in self.trading_pairs if pair in self.positions}

    async def run_backtest(self, analysis_start_date, end_date):
        pd.set_option('display.max_rows', None)
        self.usdt_balance = float(getattr(config, 'INITIAL_DEPOSIT_USDT'))

        days_since_last_reset = 0
        reset_interval = 50
        historical_data_1m, historical_data_1h = {}, {}
        tf = getattr(config, 'TIMEFRAME')
        # Dynamic lookback based on bars required for indicators
        def timeframe_to_minutes(tf_str: str) -> int:
            try:
                if tf_str.endswith('m'):
                    return int(tf_str[:-1])
                if tf_str.endswith('h'):
                    return int(tf_str[:-1]) * 60
                if tf_str.endswith('d'):
                    return int(tf_str[:-1]) * 24 * 60
            except Exception:
                pass
            return 60
        tf_minutes = timeframe_to_minutes(tf)
        bars_per_day = max(1, int((24 * 60) / tf_minutes))
        max_indicator_bars = max(200, int(getattr(config, 'DONCHIAN_PERIOD', 55)), int(getattr(config, 'ATR_PERIOD', 14)), 50)
        window_days = max(3, int(np.ceil((max_indicator_bars + 20) / bars_per_day)))
        logging.info(f"Fetching historical data timeframe={tf}, bars_per_day={bars_per_day}, window_days={window_days}")
        primary_data = await self.fetch_historical_data(analysis_start_date - datetime.timedelta(days=window_days), end_date, timeframe=tf)
        historical_data_1m = primary_data

        for pair, df in historical_data_1m.items():
            try:
                has_date = 'date' in df.columns
                is_dt = pd.api.types.is_datetime64_any_dtype(df['date']) if has_date else False
                logging.info(
                    f"run_backtest pre-check: {pair}: shape={df.shape}, has_date={has_date}, date_is_datetime={is_dt}"
                )
                if has_date and not is_dt:
                    logging.info(f"run_backtest: coercing 'date' column to datetime for {pair}")
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    logging.info(f"run_backtest post-coerce: {pair}: date_is_datetime={pd.api.types.is_datetime64_any_dtype(df['date'])}")
            except Exception as e:
                logging.error(f"run_backtest pre-check: logging error for {pair}: {e}")
        for current_date in self.generate_date_range(analysis_start_date, end_date):

            # reuse window_days computed above for indicator stability
            daily_data_1m = {
                pair: data[(data['date'] >= current_date - datetime.timedelta(days=window_days)) & (data['date'].dt.date <= current_date.date())]
                for pair, data in historical_data_1m.items()
            }
            daily_data_1h = {
                pair: data[(data['date'] >= current_date - datetime.timedelta(days=3)) & (data['date'].dt.date <= current_date.date())]
                for pair, data in historical_data_1h.items()
            }

            if daily_data_1m:
                await self.simulate_day_trading(daily_data_1m, daily_data_1h, current_date)

        await self.save_visualization_data()
        self.display_backtesting_results()

    def generate_date_range(self, start_date, end_date):
        current_date = start_date
        while current_date < end_date:
            yield current_date
            current_date += datetime.timedelta(days=1)

    async def simulate_day_trading(self, daily_data_1m, daily_data_1h, current_date):
        all_timestamps = sorted(set(
            ts for df in daily_data_1m.values() for ts in df[df['date'].dt.date >= current_date.date()]['date']
        ))
        for timestamp in all_timestamps:
            tasks = []
            for pair, data_1m in daily_data_1m.items():
                filtered_data_1m = data_1m[data_1m['date'] == timestamp]
                if not filtered_data_1m.empty:
                    current_data_slice_1m = data_1m[data_1m['date'] <= timestamp]
                    if await self.should_analyze_pair(pair):
                        task = asyncio.create_task(self.analyze_data_and_trade(pair, current_data_slice_1m, {}))
                        tasks.append(task)

            await asyncio.gather(*tasks)

    async def should_analyze_pair(self, pair):
        async with self.lock:
            open_trade = self.open_trades.get(pair)

        if open_trade:
            logging.info(f"Trade already open for {pair}. Analyzing for potential actions.")
            return True
        else:
            logging.info(f"No open trades for {pair}. Proceeding with analysis.")
            return True

    async def analyze_data_and_trade(self, pair, current_data_1m, current_data_1h):
        try:
            if not await self.should_analyze_pair(pair):
                return

            if current_data_1m.empty:
                logging.info(f"No data available for {pair} at this time.")
                return

            current_price_1m = current_data_1m['close'].iloc[-1]
            current_date_1m = current_data_1m['date'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')

            data = current_data_1m.reset_index(drop=True)
            support_levels, resistance_levels = self.find_pivot_points(data)
            async with self.lock:
                filtered_support_levels = [(time, level) for time, level in support_levels if pd.to_datetime(time).date() == current_data_1m['date'].iloc[-1].date()]
                self.visualization_data_1m[pair]['supportLevels'].extend(filtered_support_levels)

                unique_support_dict = {time: level for time, level in reversed(self.visualization_data_1m[pair]['supportLevels'])}
                self.visualization_data_1m[pair]['supportLevels'] = [(time, level) for time, level in unique_support_dict.items()]
                self.visualization_data_1m[pair]['supportLevels'].sort(key=lambda x: x[0])

                filtered_resistance_levels = [(time, level) for time, level in resistance_levels if pd.to_datetime(time).date() == current_data_1m['date'].iloc[-1].date()]
                self.visualization_data_1m[pair]['resistanceLevels'].extend(filtered_resistance_levels)

                unique_resist_dict = {time: level for time, level in reversed(self.visualization_data_1m[pair]['resistanceLevels'])}
                self.visualization_data_1m[pair]['resistanceLevels'] = [(time, level) for time, level in unique_resist_dict.items()]
                self.visualization_data_1m[pair]['resistanceLevels'].sort(key=lambda x: x[0])

            # self.update_trend_state(pair, self.visualization_data_1m[pair]['supportLevels'], self.visualization_data_1m[pair]['resistanceLevels'], current_data_1m['date'].iloc[-1])

            indicators_1m = self.calculate_indicators(current_data_1m, pair)
            if indicators_1m['current']['rsi'] < 25:
                async with self.lock:
                    self.signals[pair]['oversold'] = True

            if indicators_1m['current']['rsi'] > 75:
                async with self.lock:
                    self.signals[pair]['overbought'] = True

            buy_signal = False
            sell_signal = False

            new_row_1m = {
                'time': current_date_1m,
                'value': current_price_1m,
                'open': current_data_1m['open'].iloc[-1],
                'high': current_data_1m['high'].iloc[-1],
                'low': current_data_1m['low'].iloc[-1],
                'close': current_price_1m,
                'RSI': indicators_1m['current']['rsi'],
                'MACD': indicators_1m['current']['macd'][0],
                'Signal': indicators_1m['current']['macd'][1],
                'Histogram': indicators_1m['current']['histogram']
            }

            logging.debug(f"Current price for {pair} at {current_date_1m}: {current_price_1m}, Indicators: {indicators_1m['current']}")

            async with self.lock:
                open_trade = self.open_trades.get(pair)
            logging.debug(f"OPEN TRADE {open_trade}")

            if open_trade:
                if pair in self.positions:
                    async with self.lock:
                        logging.debug(f"POSITIONS {self.positions[pair]}")
                        take_profit_price = self.positions[pair]['take_profit_price']
                        stop_loss_price = self.positions[pair]['stop_loss_price']
                        position_type = self.positions[pair]['position_type']
                        open_price_pos = self.positions[pair]['open_price']
                        init_tp = self.positions[pair].get('initial_take_profit_price')
                        init_sl = self.positions[pair].get('initial_stop_loss_price')

                    if position_type == 'long':
                        if current_price_1m >= take_profit_price and not self.positions[pair].get('take_profit_triggered', False):
                            logging.info(f"Long TP: {pair} at {current_price_1m}")
                            sell_signal = True
                            await self.close_position(pair, current_price_1m, current_date_1m, reason='tp')
                            async with self.lock:
                                self.visualization_data_1m[pair]['tradeMarkers'].append({
                                    'time': current_date_1m,
                                    'position': 'aboveBar',
                                    'color': '#26a69a',
                                    'shape': 'arrowDown',
                                    'text': 'Long TP',
                                    'price': take_profit_price,
                                    'executed_price': current_price_1m,
                                    'entry_price': open_price_pos,
                                    'tp_price': take_profit_price,
                                    'sl_price': stop_loss_price,
                                    'initial_tp_price': init_tp,
                                    'initial_sl_price': init_sl
                                })
                            logging.debug(f"Marker: Long TP {pair} t={current_date_1m} tp={take_profit_price} sl={stop_loss_price}")
                            async with self.lock:
                                self.signals = { pair: { 'overbought': False, 'oversold': False} for pair in self.trading_pairs}
                        elif current_price_1m <= stop_loss_price and not self.positions[pair].get('stop_loss_triggered', False):
                            logging.info(f"Long SL/TS: {pair} at {current_price_1m}")
                            sell_signal = True
                            # Determine if this stop is a trailing stop before closing
                            is_trailing = stop_loss_price >= open_price_pos
                            await self.close_position(pair, current_price_1m, current_date_1m, reason=('ts' if is_trailing else 'sl'))
                            async with self.lock:
                                stop_text = 'Long TS' if is_trailing else 'Long SL'
                                stop_pos = 'aboveBar' if is_trailing else 'belowBar'
                                self.visualization_data_1m[pair]['tradeMarkers'].append({
                                    'time': current_date_1m,
                                    'position': stop_pos,
                                    'color': '#ef5350',
                                    'shape': 'arrowDown',
                                    'text': stop_text,
                                    'price': stop_loss_price,
                                    'executed_price': current_price_1m,
                                    'entry_price': open_price_pos,
                                    'tp_price': take_profit_price,
                                    'sl_price': stop_loss_price,
                                    'initial_tp_price': init_tp,
                                    'initial_sl_price': init_sl
                                })
                            logging.debug(f"Marker: Long SL/TS {pair} t={current_date_1m} sl={stop_loss_price}")
                            async with self.lock:
                                self.signals = { pair: { 'overbought': False, 'oversold': False} for pair in self.trading_pairs}
                            logging.debug(f"Long stop executed for {pair}")
                            return
                    elif position_type == 'short':
                        if current_price_1m <= take_profit_price and not self.positions[pair].get('take_profit_triggered', False):
                            logging.info(f"Short TP: {pair} at {current_price_1m}")
                            buy_signal = True
                            await self.close_position(pair, current_price_1m, current_date_1m, reason='tp')
                            async with self.lock:
                                self.visualization_data_1m[pair]['tradeMarkers'].append({
                                    'time': current_date_1m,
                                    'position': 'belowBar',
                                    'color': '#26a69a',
                                    'shape': 'arrowUp',
                                    'text': 'Short TP',
                                    'price': take_profit_price,
                                    'executed_price': current_price_1m,
                                    'entry_price': open_price_pos,
                                    'tp_price': take_profit_price,
                                    'sl_price': stop_loss_price,
                                    'initial_tp_price': init_tp,
                                    'initial_sl_price': init_sl
                                })
                            logging.debug(f"Marker: Short TP {pair} t={current_date_1m} tp={take_profit_price} sl={stop_loss_price}")
                            async with self.lock:
                                self.signals = { pair: { 'overbought': False, 'oversold': False} for pair in self.trading_pairs}
                        elif current_price_1m >= stop_loss_price and not self.positions[pair].get('stop_loss_triggered', False):
                            logging.info(f"Short SL/TS: {pair} at {current_price_1m}")
                            buy_signal = True
                            # Determine if this stop is a trailing stop before closing
                            is_trailing = stop_loss_price <= open_price_pos
                            await self.close_position(pair, current_price_1m, current_date_1m, reason=('ts' if is_trailing else 'sl'))
                            async with self.lock:
                                stop_text = 'Short TS' if is_trailing else 'Short SL'
                                stop_pos = 'belowBar' if is_trailing else 'aboveBar'
                                self.visualization_data_1m[pair]['tradeMarkers'].append({
                                    'time': current_date_1m,
                                    'position': stop_pos,
                                    'color': '#ef5350',
                                    'shape': 'arrowUp',
                                    'text': stop_text,
                                    'price': stop_loss_price,
                                    'executed_price': current_price_1m,
                                    'entry_price': open_price_pos,
                                    'tp_price': take_profit_price,
                                    'sl_price': stop_loss_price,
                                    'initial_tp_price': init_tp,
                                    'initial_sl_price': init_sl
                                })
                            logging.debug(f"Marker: Short SL/TS {pair} t={current_date_1m} sl={stop_loss_price}")
                            async with self.lock:
                                self.signals = { pair: { 'overbought': False, 'oversold': False} for pair in self.trading_pairs}
                            logging.debug(f"Short stop executed for {pair}")

            # Trailing stop adjustment if enabled and position open
            async with self.lock:
                pos = self.positions.get(pair)
            if pos and getattr(config, 'USE_TRAILING_STOP', False):
                atr_now = indicators_1m['current'].get('atr')
                if atr_now and not np.isnan(atr_now):
                    trail_mult = float(getattr(config, 'ATR_MULT_TRAIL', 2.0))
                    if pos['position_type'] == 'long':
                        # Move stop to max of current stop and last close - atr*mult
                        new_sl = max(pos['stop_loss_price'], current_price_1m - atr_now * trail_mult)
                        async with self.lock:
                            self.positions[pair]['stop_loss_price'] = new_sl
                    else:
                        new_sl = min(pos['stop_loss_price'], current_price_1m + atr_now * trail_mult)
                        async with self.lock:
                            self.positions[pair]['stop_loss_price'] = new_sl
                # Optional break-even move after price moves in favor by N ATRs
                if bool(getattr(config, 'USE_BREAK_EVEN', False)):
                    be_mult = float(getattr(config, 'BREAK_EVEN_ATR_MULT', 1.0))
                    if atr_now and not np.isnan(atr_now):
                        if pos['position_type'] == 'long':
                            if current_price_1m >= pos['open_price'] + atr_now * be_mult:
                                new_sl = max(pos['stop_loss_price'], pos['open_price'])
                                async with self.lock:
                                    self.positions[pair]['stop_loss_price'] = new_sl
                        else:
                            if current_price_1m <= pos['open_price'] - atr_now * be_mult:
                                new_sl = min(pos['stop_loss_price'], pos['open_price'])
                                async with self.lock:
                                    self.positions[pair]['stop_loss_price'] = new_sl

            trade_amount = self.calculate_trade_amount_with_leverage(current_price_1m)

            # Entry cooldown: avoid re-entering too fast
            cooldown_bars = int(getattr(config, 'ENTRY_COOLDOWN_BARS', 0))
            recently_entered = False
            if cooldown_bars > 0:
                last_trades = self.trades.get(pair, [])
                if last_trades:
                    last_close = pd.to_datetime(last_trades[-1]['close_date'])
                    # bar spacing equals current_data_1m last bar minus previous bar
                    if len(current_data_1m) >= 2:
                        bar_delta = current_data_1m['date'].iloc[-1] - current_data_1m['date'].iloc[-2]
                        cutoff = current_data_1m['date'].iloc[-1] - bar_delta * cooldown_bars
                        recently_entered = last_close > cutoff
            if open_trade is None and not recently_entered and self.should_long(indicators_1m, {}, current_price_1m):
                logging.info(f"Long signal: {pair} at {current_price_1m}")
                buy_signal = True
                await self.execute_simulated_order(pair, 'long', current_price_1m, trade_amount, current_date_1m, atr_value=indicators_1m['current'].get('atr'))
                logging.debug(f"Long order executed for {pair}.")
                # Mark long entry
                async with self.lock:
                    pos = self.positions.get(pair)
                    self.visualization_data_1m[pair]['tradeMarkers'].append({
                        'time': current_date_1m,
                        'position': 'belowBar',
                        'color': 'green',
                        'shape': 'arrowUp',
                        'text': 'Long Entry',
                        'price': current_price_1m,
                        'entry_price': pos['open_price'] if pos else current_price_1m,
                        'tp_price': pos['take_profit_price'] if pos else None,
                        'sl_price': pos['stop_loss_price'] if pos else None
                    })
                logging.debug(f"Marker: Long Entry {pair} t={current_date_1m}")
            elif open_trade is None and not recently_entered and self.should_short(indicators_1m, {}, current_price_1m) and bool(getattr(config, 'ALLOW_SHORTS', True)):
                logging.info(f"Short signal: {pair} at {current_price_1m}")
                sell_signal = True
                await self.execute_simulated_order(pair, 'short', current_price_1m, trade_amount, current_date_1m, atr_value=indicators_1m['current'].get('atr'))
                logging.debug(f"Short order executed for {pair}.")
                # Mark short entry
                async with self.lock:
                    pos = self.positions.get(pair)
                    self.visualization_data_1m[pair]['tradeMarkers'].append({
                        'time': current_date_1m,
                        'position': 'aboveBar',
                        'color': 'red',
                        'shape': 'arrowDown',
                        'text': 'Short Entry',
                        'price': current_price_1m,
                        'entry_price': pos['open_price'] if pos else current_price_1m,
                        'tp_price': pos['take_profit_price'] if pos else None,
                        'sl_price': pos['stop_loss_price'] if pos else None
                    })
                logging.debug(f"Marker: Short Entry {pair} t={current_date_1m}")
            else:
                logging.debug(f"No trade signal for {pair} at {current_price_1m}")

            # Legacy Buy/Sell flags removed; using explicit tradeMarkers instead

            async with self.lock:
                if pair not in self.visualization_data_1m:
                    self.visualization_data_1m[pair] = {'candlestickSeries': [], 'supportLevels': [], 'resistanceLevels': [], 'tradeMarkers': []}
                self.visualization_data_1m[pair]['candlestickSeries'].append(new_row_1m)

        except Exception as e:
            logging.error(f"Error analyzing pair {pair} during backtesting: {e}, Traceback: {traceback.format_exc()}")

    def find_last_local_min(self, df):
        for i in range(len(df) - 2, 0, -1):
            if df['low'].iloc[i] < df['low'].iloc[i - 1] and df['low'].iloc[i] < df['low'].iloc[i + 1]:
                return i, df['low'].iloc[i]
        return None, None

    def find_pivot_points(self, data, leftbars=4, rightbars=4):
        highs = data['high']
        lows = data['low']
        dates = data['date']
        resistance_levels = []
        support_levels = []

        last_level = None

        for i in range(leftbars, len(data) - rightbars):
            windowed_high = highs[i - leftbars:i + rightbars + 1]
            windowed_low = lows[i - leftbars:i + rightbars + 1]

            if highs[i] == max(windowed_high) and last_level != 'resistance':
                resistance_levels.append((dates.iloc[i].strftime('%Y-%m-%d %H:%M:%S'), highs[i]))
                last_level = 'resistance'

            elif lows[i] == min(windowed_low) and last_level != 'support':
                support_levels.append((dates.iloc[i].strftime('%Y-%m-%d %H:%M:%S'), lows[i]))
                last_level = 'support'

        return support_levels, resistance_levels

    async def execute_simulated_order(self, pair, position_type, price, amount, current_date, atr_value=None):
        leverage = int(getattr(config, 'LEVERAGE'))
        margin_per_unit = price / leverage
        total_margin_required = margin_per_unit * amount
        # Use ATR-based exits if enabled and indicators are available
        if getattr(config, 'USE_BREAKOUT_STRATEGY', False):
            # Try to use the last computed indicators on the current slice
            try:
                # Fallback: compute ATR on the fly is heavy; rely on recent added indicators via analyze_data_and_trade path
                # Here we approximate using multipliers around entry
                atr_mult_stop = float(getattr(config, 'ATR_MULT_STOP', 2.0))
                atr_mult_tp = float(getattr(config, 'ATR_MULT_TP', 3.0))
                # Use ATR passed from indicators if available; fallback to 0.5% of price
                assumed_atr = float(atr_value) if (atr_value is not None and not np.isnan(atr_value)) else price * 0.005
                if position_type == 'long':
                    take_profit_price = price + assumed_atr * atr_mult_tp
                    stop_loss_price = price - assumed_atr * atr_mult_stop
                else:
                    take_profit_price = price - assumed_atr * atr_mult_tp
                    stop_loss_price = price + assumed_atr * atr_mult_stop
            except Exception:
                if position_type == 'long':
                    take_profit_price = price * (1 + config.TAKE_PROFIT_PERCENTAGE)
                    stop_loss_price = price * (1 - config.STOP_LOSS_PERCENTAGE)
                else:
                    take_profit_price = price * (1 - config.TAKE_PROFIT_PERCENTAGE)
                    stop_loss_price = price * (1 + config.STOP_LOSS_PERCENTAGE)
        else:
            if position_type == 'long':
                take_profit_price = price * (1 + config.TAKE_PROFIT_PERCENTAGE)
                stop_loss_price = price * (1 - config.STOP_LOSS_PERCENTAGE)
            elif position_type == 'short':
                take_profit_price = price * (1 - config.TAKE_PROFIT_PERCENTAGE)
                stop_loss_price = price * (1 + config.STOP_LOSS_PERCENTAGE)

        # Enforce invariant at open: long => SL < entry < TP; short => TP < entry < SL
        epsilon = max(1e-8, price * 1e-6)
        orig_tp, orig_sl = take_profit_price, stop_loss_price
        if position_type == 'long':
            if stop_loss_price >= price:
                stop_loss_price = price - epsilon
            if take_profit_price <= price:
                take_profit_price = price + epsilon
        else:
            if stop_loss_price <= price:
                stop_loss_price = price + epsilon
            if take_profit_price >= price:
                take_profit_price = price - epsilon
        if orig_tp != take_profit_price or orig_sl != stop_loss_price:
            logging.warning(f"Adjusted initial TP/SL to satisfy invariants: type={position_type} entry={price} tp:{orig_tp}->{take_profit_price} sl:{orig_sl}->{stop_loss_price}")

        if self.usdt_balance >= total_margin_required:
            self.usdt_balance -= total_margin_required

            if position_type == 'long':
                logging.info(f"Opened long position: {amount} of {pair} at {price}, using ${total_margin_required} as margin")
            elif position_type == 'short':
                logging.info(f"Opened short position: {amount} of {pair} at {price}, using ${total_margin_required} as margin")

            async with self.lock:
                self.positions[pair] = {
                    'position_type': position_type,
                    'amount': amount,
                    'open_price': price,
                    'margin_used': total_margin_required,
                    'open_date': current_date,
                    'take_profit_price': take_profit_price,
                    'take_profit_triggered': False,
                    'stop_loss_price': stop_loss_price,
                    'stop_loss_triggered': False,
                    'initial_take_profit_price': take_profit_price,
                    'initial_stop_loss_price': stop_loss_price,
                }
                self.open_trades[pair] = {'pair': pair, 'status': 'open', 'price': price, 'open_date': current_date, 'position_type': position_type}
            logging.info(f"Order SL/TP set: pair={pair} type={position_type} entry={price} tp={take_profit_price} sl={stop_loss_price} atr={assumed_atr if 'assumed_atr' in locals() else 'n/a'}")
        else:
            logging.warning(f"Insufficient balance to open position. Required margin: ${total_margin_required}, Available balance: ${self.usdt_balance}")

    async def close_position(self, pair, close_price, current_date, reason: str = None):
        if pair in self.positions:
            async with self.lock:
                open_trade = self.positions[pair]
            close_trade = {
                'close_price': close_price,
                'close_date': current_date,
                'amount': open_trade['amount'],
                'position_type': open_trade['position_type']
            }
            close_trade['exit_reason'] = reason or 'unknown'
            profit_details = self.calculate_pair_profit(pair, open_trade, close_trade)

            # Win/Loss/Break-even classification log with details
            net_pnl = profit_details.get('net_profit', 0.0)
            open_fee = profit_details.get('open_fee', 0.0)
            close_fee = profit_details.get('close_fee', 0.0)
            amount = close_trade['amount']
            entry = open_trade['open_price']
            exit_ = close_trade['close_price']
            pos_type = open_trade['position_type']
            result = 'WIN' if net_pnl > 0 else ('LOSS' if net_pnl < 0 else 'BE')
            logging.info(
                f"Trade closed [{result}] {pair} pos={pos_type} entry={entry} exit={exit_} "
                f"amount={amount} pnl={net_pnl:.4f} fees(open={open_fee:.4f}, close={close_fee:.4f}) reason={close_trade['exit_reason']}"
            )

            async with self.lock:
                del self.open_trades[pair]
                del self.positions[pair]

            logging.debug(f"Position closed for {pair}. Profit: {profit_details['net_profit']}")
            logging.debug(f"Balance is ${self.usdt_balance}.")
        else:
            logging.warning(f"No open position found for {pair}")

    def calculate_pair_profit(self, pair, open_trade, close_trade):
        position_type = open_trade['position_type']
        open_price = open_trade['open_price']
        close_price = close_trade['close_price']
        amount = open_trade['amount']
        open_date = open_trade['open_date']
        close_date = close_trade['close_date']
        margin_used = open_trade['margin_used']
        logging.debug(f"Open price: {open_price}, Close price: {close_price}, Amount: {amount}, Open date: {open_date}, Close date: {close_date}")
        fee_rate = config.TRADING_FEES.get(self.client.name, 0.001)
        open_fee = open_price * amount * fee_rate
        close_fee = close_price * amount * fee_rate
        logging.debug(f"Position type: {position_type}, Open fee: {open_fee}, Close fee: {close_fee}")
        if position_type == 'long':
            cost = open_price * amount + open_fee
            revenue = close_price * amount - close_fee
            logging.debug(f"Cost: {cost}")
            logging.debug(f"Revenue: {revenue}")
        elif position_type == 'short':
            revenue = open_price * amount - open_fee
            cost = close_price * amount + close_fee
            logging.debug(f"Cost: {cost}")
            logging.debug(f"Revenue: {revenue}")

        net_profit = revenue - cost
        self.usdt_balance += net_profit + margin_used

        self.log_trade(
            pair,
            open_price,
            close_price,
            amount,
            open_fee,
            close_fee,
            net_profit,
            open_date,
            close_date,
            exit_reason=close_trade.get('exit_reason')
        )

        return {
            'net_profit': net_profit,
            'open_fee': open_fee,
            'close_fee': close_fee,
            'revenue': revenue,
            'cost': cost
        }

    def log_trade(self, pair, buy_price, sell_price, amount, buy_fee, sell_fee, profit, open_date, close_date, exit_reason=None):
        trade = {
            'pair': pair,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'buy_fee': buy_fee,
            'sell_fee': sell_fee,
            'amount': amount,
            'profit': profit,
            'open_date': open_date,
            'close_date': close_date,
            'exit_reason': exit_reason
        }
        logging.info(f"Trade info: {trade}")
        if pair not in self.trades:
            self.trades[pair] = []

        self.trades[pair].append(trade)

    def calculate_trade_amount_with_leverage(self, price):
        leverage = int(getattr(config, 'LEVERAGE', 25))
        risk = float(getattr(config, 'RISK_PER_TRADE_PCT', 0.01))
        amount = self.usdt_balance * risk
        if self.usdt_balance >= amount:
            return (amount * leverage) / price
        return 0

    def calculate_indicators(self, data, pair):
        sma_short_period = 20
        sma_long_period = 50

        sma_short = talib.SMA(data['close'].values, timeperiod=sma_short_period)
        sma_long = talib.SMA(data['close'].values, timeperiod=sma_long_period)

        rsi = talib.RSI(data['close'].values, timeperiod=14)
        macd, signal_line, histogram = talib.MACD(data['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
        previous_close = data['close'].iloc[-2] if len(data) > 1 else None

        adx_length = 14
        adx = talib.ADX(data['high'].values, data['low'].values, data['close'].values, timeperiod=adx_length)

        volume_ma = talib.SMA(data['volume'].values, timeperiod=20)

        upper_band, middle_band, lower_band = talib.BBANDS(data['close'].values, timeperiod=20)
        sma200 = talib.SMA(data['close'].values, timeperiod=200)

        # Donchian channels (highest high / lowest low over period)
        donchian_period = int(getattr(config, 'DONCHIAN_PERIOD', 20))
        donchian_upper = pd.Series(data['high']).rolling(window=donchian_period).max().values
        donchian_lower = pd.Series(data['low']).rolling(window=donchian_period).min().values

        # ATR
        atr_period = int(getattr(config, 'ATR_PERIOD', 14))
        atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=atr_period)

        supports = self.find_supports(data)
        resistances = self.find_resistances(data)

        current_support = supports.iloc[-1] if not supports.empty else None
        current_resistance = resistances.iloc[-1] if not resistances.empty else None

        sma_short_previous = talib.SMA(data['close'].values[:-1], timeperiod=sma_short_period)
        sma_long_previous = talib.SMA(data['close'].values[:-1], timeperiod=sma_long_period)
        rsi_previous = talib.RSI(data['close'].values[:-1], timeperiod=14)
        macd_previous, signal_line_previous, histogram_previous = talib.MACD(data['close'].values[:-1], fastperiod=12, slowperiod=26, signalperiod=9)
        volume_ma_previous = talib.SMA(data['volume'].values[:-1], timeperiod=sma_short_period)

        return {
            'pair': pair,
            'current': {
                'sma_short': sma_short[-1],
                'sma_long': sma_long[-1],
                'rsi': rsi[-1],
                'rsi_history': rsi[-6:],
                'macd': (macd[-1], signal_line[-1]),
                'adx': adx[-1],
                'previous_close': previous_close,
                'close': data['close'].iloc[-1],
                'histogram': histogram[-1],
                'support': current_support,
                'resistance': current_resistance,
                'volume': data['volume'].iloc[-1],
                'volume_ma': volume_ma[-1],
                'middle_band': middle_band[-1],
                'sma200': sma200[-1],
                'donchian_upper': donchian_upper[-1],
                'donchian_lower': donchian_lower[-1],
                'atr': atr[-1]
            },
            'previous': {
                'sma_short': sma_short_previous[-1],
                'sma_long': sma_long_previous[-1],
                'rsi': rsi_previous[-1],
                'rsi_history': rsi_previous[-6:],
                'macd': (macd_previous[-1], signal_line_previous[-1]),
                'close': data['close'].iloc[-2],
                'volume_ma': volume_ma_previous[-1],
                'donchian_upper': donchian_upper[-2] if len(donchian_upper) > 1 else np.nan,
                'donchian_lower': donchian_lower[-2] if len(donchian_lower) > 1 else np.nan,
                'atr': atr[-2] if len(atr) > 1 else np.nan
            }
        }

    def find_supports(self, data, window=20):
        minima = data['low'].rolling(window=window, center=True).min()
        is_support = data['low'] == minima
        supports = data['low'][is_support]
        return supports

    def find_resistances(self, data, window=20):
        maxima = data['high'].rolling(window=window, center=True).max()
        is_resistance = data['high'] == maxima
        resistances = data['high'][is_resistance]
        return resistances

    async def save_visualization_data(self):
        for pair, data in self.visualization_data_1m.items():
            filename = f'data/{pair.replace("/", "_")}_visualization_data.json'
            try:
                logging.debug(f"Saving visualization for {pair}: candles={len(data.get('candlestickSeries', []))} markers={len(data.get('tradeMarkers', []))}")
            except Exception:
                logging.debug(f"Saving visualization for {pair}")
            with open(filename, 'w') as file:
                json.dump(data, file, indent=4)

    def display_backtesting_results(self):
        total_all_pairs_trades = 0
        total_all_winning_trades = 0
        total_all_losing_trades = 0
        for pair, trades in self.trades.items():
            total_trades = len(trades)
            total_all_pairs_trades += total_trades
            winning_trades = [trade for trade in trades if trade['profit'] > 0]
            losing_trades = [trade for trade in trades if trade['profit'] < 0]

            total_net_profit = sum(trade['profit'] for trade in trades if trade['profit'] > 0)
            total_loss = sum(trade['profit'] for trade in trades if trade['profit'] < 0)

            average_profit = sum(trade['profit'] for trade in winning_trades) / len(winning_trades) if winning_trades else 0
            average_loss = sum(-trade['profit'] for trade in losing_trades) / len(losing_trades) if losing_trades else 0

            # Breakdown by exit reason
            ts_trades = [t for t in trades if (t.get('exit_reason') == 'ts')]
            tp_trades = [t for t in trades if (t.get('exit_reason') == 'tp')]
            sl_trades = [t for t in trades if (t.get('exit_reason') == 'sl')]
            ts_profit = sum(t['profit'] for t in ts_trades)
            tp_profit = sum(t['profit'] for t in tp_trades)
            sl_profit = sum(t['profit'] for t in sl_trades)

            total_all_winning_trades += len(winning_trades)
            total_all_losing_trades += len(losing_trades)

            logging.info(f"Backtesting Results for {pair}:")
            logging.info(f"Total Trades: {total_trades}")
            logging.info(f"Winning Trades: {len(winning_trades)}")
            logging.info(f"Losing Trades: {len(losing_trades)}")
            logging.info(f"Total Net Profit: {total_net_profit:.2f} USDT")
            logging.info(f"Total Loss: {total_loss:.2f} USDT")
            logging.info(f"Average Profit per Winning Trade: {average_profit:.2f} USDT")
            logging.info(f"Average Loss per Losing Trade: {average_loss:.2f} USDT")
            logging.info(f"TS Trades: {len(ts_trades)} | TS PnL: {ts_profit:.2f} USDT")
            logging.info("-" * 50)

        logging.info(f"Total Number of Trades Across All Pairs: {total_all_pairs_trades}")
        logging.info(f"Total Winning Trades Across All Pairs: {total_all_winning_trades}")
        logging.info(f"Total Losing Trades Across All Pairs: {total_all_losing_trades}")

        if self.positions:
            logging.info("Open Positions (locked margin):")
            total_locked_margin = 0.0
            for pair, pos in self.positions.items():
                locked = float(pos.get('margin_used', 0.0))
                total_locked_margin += locked
                logging.info(f" - {pair}: {locked:.4f} USDT")
        else:
            total_locked_margin = 0.0

        total_equity = self.usdt_balance + total_locked_margin
        logging.info(f"Free Balance: {self.usdt_balance:.4f} USDT")
        logging.info(f"Locked Margin: {total_locked_margin:.4f} USDT")
        logging.info(f"Equity (Free + Locked): {total_equity:.4f} USDT")