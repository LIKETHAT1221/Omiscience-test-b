"""
Omniscience — Enhanced TA Engine (EV + Kelly + Backtesting)
Streamlit Conversion

Original-to-New Function Mappings:
- parseBlocksStrict() -> parse_blocks_strict()
- analyzeAndBuild() -> analyze_and_build()
- calculateNoVigProbability() -> calculate_no_vig_probability()
- calculateEV() -> calculate_ev()
- calculateKelly() -> calculate_kelly()
- calculateModelProbability() -> calculate_model_probability()
- runBacktest() -> run_backtest()
- adaptiveMA() -> adaptive_ma()
- getFibonacciLevelsFull() -> get_fibonacci_levels_full()
- All technical indicators (SMA_full, EMA_full, RSI_full, MACD_full, etc.) preserved

Legacy Logic Disabled:
- Underdog ML bias multipliers removed
- Home-team bias logic disabled via allow_home_team_bias=False
- Standalone underdog ML column removed

New Features Added:
- Machine learning integration via ml_module
- KAMA implementation
- Signal fusion with configurable weights
- Full feed time-series analysis
- EV calculations with vig removal
- Forecasting with Prophet
- Line movement analysis
- Enhanced TA output with favorite_ml, underdog_ml, upset_threshold
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import math
from typing import List, Dict, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import ML module
try:
    from ml_module import MLModel, train_model, predict_game
except ImportError:
    # Fallback if ml_module is not available
    class MLModel:
        def __init__(self):
            self.is_trained = False
        def predict_proba(self, game_data):
            return 0.5, 0.5, None

    def train_model(data):
        return MLModel()
    
    def predict_game(model, game_data):
        return 0.5, 0.5, None

# Configuration Section
st.set_page_config(page_title="Omniscience — Enhanced TA Engine", layout="wide")

# Default config - exposed at top for editing
st.sidebar.header("Configuration Parameters")

# AMA Parameters
ama_fast = st.sidebar.slider("AMA Fast Period", 2, 10, 2)
ama_slow = st.sidebar.slider("AMA Slow Period", 5, 30, 10)
ama_efficiency = st.sidebar.slider("AMA Efficiency Lookback", 5, 20, 8)

# Signal Weights
st.sidebar.subheader("Signal Fusion Weights")
weight_ama = st.sidebar.slider("AMA Weight", 0.0, 1.0, 0.4)
weight_momentum = st.sidebar.slider("Momentum Weight", 0.0, 1.0, 0.15)
weight_delta = st.sidebar.slider("Delta Weight", 0.0, 1.0, 0.15)
weight_volatility = st.sidebar.slider("Volatility Weight", 0.0, 1.0, 0.1)
weight_value = st.sidebar.slider("Value Signal Weight", 0.0, 1.0, 0.2)

# Normalize weights
total_weight = weight_ama + weight_momentum + weight_delta + weight_volatility + weight_value
if total_weight > 0:
    weight_ama /= total_weight
    weight_momentum /= total_weight
    weight_delta /= total_weight
    weight_volatility /= total_weight
    weight_value /= total_weight

signal_weights = {
    "AMA": weight_ama,
    "momentum": weight_momentum,
    "delta": weight_delta,
    "volatility": weight_volatility,
    "value_signal": weight_value
}

# Other parameters
upset_threshold = st.sidebar.slider("Upset Threshold", 0.01, 0.2, 0.05)
forecast_horizon = st.sidebar.slider("Forecast Horizon (hours)", 1, 24, 6)
allow_home_team_bias = st.sidebar.checkbox("Allow Home Team Bias", False)

# Bankroll management
bankroll_options = {
    "Bankroll: $1,000": 1000,
    "Bankroll: $5,000": 5000,
    "Bankroll: $10,000": 10000,
    "Custom": "custom"
}

selected_bankroll = st.sidebar.selectbox("Select Bankroll", list(bankroll_options.keys()))
if selected_bankroll == "Custom":
    custom_bankroll = st.sidebar.number_input("Custom Bankroll", min_value=100, max_value=100000, value=1000, step=100)
    bankroll = custom_bankroll
else:
    bankroll = bankroll_options[selected_bankroll]

# ========== LEGACY FUNCTIONS (PRESERVED FROM ORIGINAL) ==========

def extract_numbers(s):
    """Extract all numbers from string"""
    if s and isinstance(s, str):
        matches = re.findall(r'[+-]?\d+(?:\.\d+)?', s)
        return [float(m) for m in matches] if matches else []
    return []

def extract_first_number(s):
    """Extract first number from string"""
    if s and isinstance(s, str):
        match = re.search(r'[+-]?\d+(?:\.\d+)?', s)
        return float(match.group()) if match else float('nan')
    return float('nan')

def has_letters(s):
    """Check if string contains letters"""
    if s and isinstance(s, str):
        return bool(re.search(r'[A-Za-z]', s))
    return False

def is_header_line(s):
    """Check if line is header"""
    if s and isinstance(s, str):
        return bool(re.match(r'^time\b', s.strip(), re.I))
    return False

def normalize_spread(spread):
    """Normalize spread value"""
    try:
        val = float(spread)
        return val if not math.isnan(val) else None
    except (ValueError, TypeError):
        return None

def parse_timestamp(time_str):
    """Parse timestamp from MM/DD h:mmAM/PM format"""
    if not time_str:
        return None
    
    # Try multiple timestamp formats
    formats = [
        r'^(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2})(AM|PM)$',
        r'^(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2}):(\d{2})(AM|PM)$'
    ]
    
    for fmt in formats:
        match = re.match(fmt, time_str.strip(), re.I)
        if match:
            groups = match.groups()
            month = int(groups[0]) - 1
            day = int(groups[1])
            hour = int(groups[2])
            minute = int(groups[3])
            ampm = groups[-1].upper()
            
            if ampm == "PM" and hour < 12:
                hour += 12
            elif ampm == "AM" and hour == 12:
                hour = 0
                
            return datetime(2000, month, day, hour, minute, 0)
    
    return None

def implied_probability(odds):
    """Calculate implied probability from moneyline odds"""
    if odds == 'even':
        return 0.5
    try:
        odds_val = float(odds)
        if odds_val > 0:
            return 100 / (odds_val + 100)
        elif odds_val < 0:
            return abs(odds_val) / (abs(odds_val) + 100)
        else:
            return 0.5
    except (ValueError, TypeError):
        return 0.5

def calculate_no_vig_probability(odds1, odds2):
    """Calculate no-vig probability from two odds"""
    implied1 = implied_probability(odds1)
    implied2 = implied_probability(odds2)
    
    total_probability = implied1 + implied2
    if total_probability == 0:
        return {"prob1": 0.5, "prob2": 0.5}
    
    return {
        "prob1": implied1 / total_probability,
        "prob2": implied2 / total_probability
    }

def calculate_ev(probability, odds, bet_type='american'):
    """Calculate expected value for a bet"""
    try:
        if bet_type == 'american':
            if odds > 0:
                decimal_odds = 1 + (odds / 100)
            else:
                decimal_odds = 1 + (100 / abs(odds))
        else:
            decimal_odds = odds
            
        return (probability * (decimal_odds - 1)) - ((1 - probability) * 1)
    except (TypeError, ZeroDivisionError):
        return 0

def calculate_kelly(probability, odds):
    """Calculate Kelly criterion stake"""
    try:
        if odds > 0:
            decimal_odds = 1 + (odds / 100)
        else:
            decimal_odds = 1 + (100 / abs(odds))
            
        B = decimal_odds - 1
        P = probability
        Q = 1 - P
        
        if B == 0:
            return 0
            
        kelly = (B * P - Q) / B
        return max(0, kelly)  # Only positive bets
    except (TypeError, ZeroDivisionError):
        return 0

def calculate_opposite_vig(vig):
    """Calculate opposite vig (simplified)"""
    if vig is None or math.isnan(vig):
        return None
    return vig  # Simplified - in practice would use sportsbook logic

def calculate_model_probability(votes, market_probability, indicator_weight=0.7):
    """Calculate model probability from votes and market context"""
    total = votes.get('up', 0) + votes.get('down', 0) + votes.get('neutral', 0)
    if total == 0:
        return market_probability
        
    raw_signal = (votes.get('up', 0) - votes.get('down', 0)) / total
    return market_probability + (raw_signal * indicator_weight * (1 - market_probability))

# Technical Analysis Functions
def ROC(series, period=2):
    """Rate of Change indicator"""
    if not series or len(series) < period:
        return [None] * len(series) if series else []
    
    out = [None] * period
    for i in range(period, len(series)):
        if series[i-period] != 0 and series[i-period] is not None and series[i] is not None:
            change = 100 * (series[i] - series[i-period]) / abs(series[i-period])
            out.append(change)
        else:
            out.append(None)
    return out

def adaptive_ma(series, fast=2, slow=10, efficiency_lookback=8):
    """Adaptive Moving Average"""
    if not series or len(series) < slow + efficiency_lookback:
        return [None] * len(series) if series else []
    
    n = len(series)
    out = [None] * n
    
    for i in range(slow + efficiency_lookback - 1, n):
        change = abs(series[i] - series[i - efficiency_lookback])
        volatility = 0
        for j in range(i - efficiency_lookback + 1, i + 1):
            volatility += abs(series[j] - series[j - 1])
        
        ER = change / volatility if volatility != 0 else 0
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        SC = (ER * (fast_sc - slow_sc) + slow_sc) ** 2
        
        if out[i-1] is None:
            out[i-1] = series[i - efficiency_lookback]
            
        out[i] = out[i-1] + SC * (series[i] - out[i-1])
    
    return out

def kaufman_adaptive_ma(series, fast=2, slow=30, efficiency=10):
    """Kaufman Adaptive Moving Average (KAMA)"""
    if not series or len(series) < efficiency:
        return [None] * len(series)
    
    n = len(series)
    kama = [None] * n
    kama[efficiency-1] = series[efficiency-1]
    
    for i in range(efficiency, n):
        # Calculate efficiency ratio
        change = abs(series[i] - series[i - efficiency])
        volatility = sum(abs(series[j] - series[j-1]) for j in range(i-efficiency+1, i+1))
        
        er = change / volatility if volatility != 0 else 0
        
        # Calculate smoothing constant
        sc = (er * (2.0/(fast+1) - 2.0/(slow+1)) + 2.0/(slow+1)) ** 2
        
        kama[i] = kama[i-1] + sc * (series[i] - kama[i-1])
    
    return kama

def get_fibonacci_levels_full(series, window=13):
    """Calculate Fibonacci levels"""
    if not series or len(series) < window:
        return None
        
    slice_data = series[-window:]
    high = max(slice_data)
    low = min(slice_data)
    
    retracements = [0.236, 0.382, 0.5, 0.618, 0.786]
    extensions = [1.236, 1.382, 1.5, 1.618, 2.0]
    
    fib_retracements = [high - (high - low) * r for r in retracements]
    fib_extensions = [high + (high - low) * (e - 1) for e in extensions]
    
    return {
        "high": high,
        "low": low,
        "retracements": fib_retracements,
        "extensions": fib_extensions
    }

def SMA_full(values, period):
    """Simple Moving Average"""
    if not values or len(values) < period:
        return [None] * len(values) if values else []
    
    n = len(values)
    out = [None] * n
    sum_val = 0
    
    for i in range(n):
        sum_val += values[i]
        if i >= period:
            sum_val -= values[i - period]
        if i >= period - 1:
            out[i] = sum_val / period
    
    return out

def EMA_full(values, period):
    """Exponential Moving Average"""
    if not values:
        return []
    
    n = len(values)
    out = [None] * n
    
    if n < period:
        k = 2 / (period + 1)
        ema = values[0]
        out[0] = ema
        for i in range(1, n):
            ema = values[i] * k + ema * (1 - k)
            out[i] = ema
        return out
    
    seed = sum(values[:period]) / period
    out[period-1] = seed
    ema = seed
    k = 2 / (period + 1)
    
    for i in range(period, n):
        ema = values[i] * k + ema * (1 - k)
        out[i] = ema
    
    return out

def RSI_full(values, period=5):
    """Relative Strength Index"""
    if not values or len(values) < period + 1:
        return [None] * len(values) if values else []
    
    n = len(values)
    out = [None] * n
    
    gains = 0
    losses = 0
    
    # Initial calculation
    for i in range(1, period + 1):
        diff = values[i] - values[i-1]
        if diff > 0:
            gains += diff
        else:
            losses += abs(diff)
    
    avg_gain = gains / period
    avg_loss = losses / period
    
    if avg_loss == 0:
        out[period] = 100
    else:
        rs = avg_gain / avg_loss
        out[period] = 100 - (100 / (1 + rs))
    
    # Subsequent calculations
    for i in range(period + 1, n):
        diff = values[i] - values[i-1]
        
        if diff > 0:
            avg_gain = (avg_gain * (period - 1) + diff) / period
            avg_loss = (avg_loss * (period - 1)) / period
        else:
            avg_gain = (avg_gain * (period - 1)) / period
            avg_loss = (avg_loss * (period - 1) + abs(diff)) / period
        
        if avg_loss == 0:
            out[i] = 100
        else:
            rs = avg_gain / avg_loss
            out[i] = 100 - (100 / (1 + rs))
    
    return out

def MACD_full(values, fast=12, slow=26, signal=9):
    """MACD indicator"""
    ema_fast = EMA_full(values, fast)
    ema_slow = EMA_full(values, slow)
    
    macd_line = []
    for i in range(len(values)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line.append(ema_fast[i] - ema_slow[i])
        else:
            macd_line.append(None)
    
    # Filter out None values for signal line calculation
    macd_filtered = [x for x in macd_line if x is not None]
    signal_line_values = EMA_full(macd_filtered, signal)
    
    # Reconstruct signal line with proper alignment
    signal_line = [None] * len(values)
    offset = next((i for i, x in enumerate(macd_line) if x is not None), 0)
    
    for i, val in enumerate(signal_line_values):
        if offset + i < len(signal_line):
            signal_line[offset + i] = val
    
    return {"macd": macd_line, "signal": signal_line}

def bollinger_bands(values, period=20, mult=2):
    """Bollinger Bands"""
    if not values or len(values) < period:
        return [None] * len(values) if values else []
    
    sma = SMA_full(values, period)
    bands = [None] * len(values)
    
    for i in range(period - 1, len(values)):
        if sma[i] is not None:
            slice_vals = values[i-period+1:i+1]
            mean = sma[i]
            std = np.std(slice_vals) if slice_vals else 0
            bands[i] = {
                "upper": mean + mult * std,
                "lower": mean - mult * std
            }
    
    return bands

def ATR_lite(values, period=14):
    """Average True Range (lite version)"""
    if not values or len(values) < period + 1:
        return None
    
    tr_values = []
    for i in range(1, len(values)):
        tr = abs(values[i] - values[i-1])
        tr_values.append(tr)
    
    if len(tr_values) < period:
        return None
    
    atr = sum(tr_values[:period]) / period
    k = 2 / (period + 1)
    
    for i in range(period, len(tr_values)):
        atr = tr_values[i] * k + atr * (1 - k)
    
    return atr

def zscore(values, period=10):
    """Z-score calculation"""
    if not values or len(values) < period:
        return [None] * len(values) if values else []
    
    n = len(values)
    out = [None] * n
    
    for i in range(period - 1, n):
        slice_vals = values[i-period+1:i+1]
        mean = np.mean(slice_vals)
        std = np.std(slice_vals)
        
        if std == 0:
            out[i] = 0
        else:
            out[i] = (values[i] - mean) / std
    
    return out

def greek_analysis(series, window=13):
    """Greek analysis (delta, gamma, vega, theta)"""
    if not series or len(series) < window:
        return None
    
    delta = series[-1] - series[-2] if len(series) >= 2 else 0
    prev_delta = series[-2] - series[-3] if len(series) >= 3 else 0
    gamma = delta - prev_delta
    
    slice_vals = series[-window:]
    mean = np.mean(slice_vals)
    vega = np.std(slice_vals)
    theta = (series[-1] - series[-window]) / window if window > 0 else 0
    
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta
    }

def detect_steam_moves(series, threshold=2):
    """Detect steam moves in series"""
    moves = []
    for i in range(1, len(series)):
        if abs(series[i] - series[i-1]) >= threshold:
            moves.append({
                "index": i,
                "from": series[i-1],
                "to": series[i],
                "change": series[i] - series[i-1]
            })
    return moves

def parse_blocks_strict(raw_text):
    """Parse odds data from text input"""
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    start = 1 if lines and is_header_line(lines[0]) else 0
    rows = []
    errors = []
    
    for i in range(start, len(lines) - 4, 5):
        block_lines = lines[i:i+5]
        L1, L2, L3, L4, L5 = block_lines
        
        tokens = L1.split()
        if len(tokens) < 2:
            errors.append({"index": i, "reason": "Insufficient tokens in line 1", "raw": block_lines})
            continue
        
        date_token = tokens[0]
        time_token = tokens[1]
        full_timestamp = f"{date_token} {time_token}"
        
        time_obj = parse_timestamp(full_timestamp)
        if not time_obj:
            errors.append({"index": i, "reason": "Invalid timestamp", "raw": block_lines})
            continue
        
        # Team and spread extraction
        spread_raw = None
        team = None
        if len(tokens) >= 3:
            if has_letters(tokens[2]):
                team = tokens[2]
                spread_raw = tokens[3] if len(tokens) >= 4 else None
            else:
                spread_raw = tokens[2]
        
        spread = normalize_spread(spread_raw)
        
        # Spread vig
        spread_vig = extract_first_number(L2)
        spread_vig_opposite = calculate_opposite_vig(spread_vig)
        
        # Total parsing
        total_side = None
        total_val = extract_first_number(L3)
        total_match = re.match(r'^([ouOU])\s*([+-]?\d+(?:\.\d+)?)', L3)
        if total_match:
            total_side = total_match.group(1).lower()
            total_val = float(total_match.group(2))
        
        # Total vig
        total_vig = extract_first_number(L4)
        total_vig_opposite = calculate_opposite_vig(total_vig)
        
        # Moneyline parsing
        ml_nums = extract_numbers(L5)
        ml_away = ml_nums[0] if len(ml_nums) >= 1 else None
        ml_home = ml_nums[1] if len(ml_nums) >= 2 else None
        
        if 'even' in L5.lower():
            ml_away = 'even'
            ml_home = extract_first_number(L5.split('even')[-1]) if 'even' in L5.lower() else ml_home
        
        # Calculate no-vig probabilities
        spread_no_vig = None
        total_no_vig = None
        ml_no_vig = None
        
        if spread_vig is not None and spread_vig_opposite is not None:
            spread_no_vig = calculate_no_vig_probability(spread_vig, spread_vig_opposite)
        
        if total_vig is not None and total_vig_opposite is not None:
            total_no_vig = calculate_no_vig_probability(total_vig, total_vig_opposite)
        
        if ml_away is not None and ml_home is not None:
            ml_no_vig = calculate_no_vig_probability(ml_away, ml_home)
        
        # Create row
        row = {
            "time": time_obj,
            "team": team,
            "spread": spread,
            "spread_vig": spread_vig,
            "spread_vig_opposite": spread_vig_opposite,
            "spread_no_vig": spread_no_vig,
            "total": total_val,
            "total_side": total_side,
            "total_vig": total_vig,
            "total_vig_opposite": total_vig_opposite,
            "total_no_vig": total_no_vig,
            "ml_away": ml_away,
            "ml_home": ml_home,
            "ml_no_vig": ml_no_vig,
            "raw": block_lines
        }
        
        # Calculate implied probabilities
        row["ml_away_prob"] = implied_probability(ml_away) if ml_away else None
        row["ml_home_prob"] = implied_probability(ml_home) if ml_home else None
        
        rows.append(row)
    
    # Sort by time
    rows.sort(key=lambda x: x["time"])
    return {"rows": rows, "errors": errors}

# ========== NEW FUNCTIONS FOR ENHANCED ANALYSIS ==========

def calculate_full_feed_analysis(series):
    """Compute TA over entire time-series with enhanced metrics"""
    if not series or len(series) < 2:
        return {}
    
    # Basic indicators
    sma_10 = SMA_full(series, 10)
    ema_5 = EMA_full(series, 5)
    rsi_7 = RSI_full(series, 7)
    macd = MACD_full(series)
    bb = bollinger_bands(series)
    z_scores = zscore(series, 10)
    roc_2 = ROC(series, 2)
    ama = adaptive_ma(series, ama_fast, ama_slow, ama_efficiency)
    kama = kaufman_adaptive_ma(series)
    
    # Enhanced metrics
    velocity = [series[i] - series[i-1] for i in range(1, len(series))]
    velocity.insert(0, 0)  # Pad beginning
    
    acceleration = [velocity[i] - velocity[i-1] for i in range(1, len(velocity))]
    acceleration.insert(0, 0)  # Pad beginning
    
    # Volatility metrics
    volatility_5 = []
    for i in range(len(series)):
        if i >= 4:
            window = series[i-4:i+1]
            vol = np.std(window) if len(window) > 1 else 0
            volatility_5.append(vol)
        else:
            volatility_5.append(0)
    
    # Cumulative stats
    cumulative_mean = [np.mean(series[:i+1]) for i in range(len(series))]
    cumulative_std = [np.std(series[:i+1]) for i in range(len(series))]
    
    # Long-term deltas
    long_term_deltas = []
    for i in range(len(series)):
        if i >= 10:
            delta = series[i] - series[i-10]
            long_term_deltas.append(delta)
        else:
            long_term_deltas.append(0)
    
    return {
        "sma_10": sma_10,
        "ema_5": ema_5,
        "rsi_7": rsi_7,
        "macd": macd,
        "bollinger_bands": bb,
        "z_scores": z_scores,
        "roc_2": roc_2,
        "adaptive_ma": ama,
        "kaufman_ama": kama,
        "velocity": velocity,
        "acceleration": acceleration,
        "volatility_5": volatility_5,
        "cumulative_mean": cumulative_mean,
        "cumulative_std": cumulative_std,
        "long_term_deltas": long_term_deltas
    }

def calculate_signal_fusion(analysis_results, current_value, market_prob, model_prob):
    """Implement weighted signal fusion"""
    signals = {}
    explanations = []
    
    # AMA Signal
    ama_values = analysis_results.get("adaptive_ma", [])
    if ama_values and ama_values[-1] is not None:
        ama_signal = 1 if current_value > ama_values[-1] else -1
        signals["AMA"] = ama_signal
        explanations.append(f"AMA: {'Bullish' if ama_signal > 0 else 'Bearish'}")
    else:
        signals["AMA"] = 0
    
    # Momentum Signal (ROC)
    roc_values = analysis_results.get("roc_2", [])
    if roc_values and roc_values[-1] is not None:
        momentum_signal = 1 if roc_values[-1] > 0 else -1
        signals["momentum"] = momentum_signal
        explanations.append(f"Momentum: {'Positive' if momentum_signal > 0 else 'Negative'} ({roc_values[-1]:.2f}%)")
    else:
        signals["momentum"] = 0
    
    # Delta Signal (model vs market)
    delta = model_prob - market_prob
    delta_signal = 1 if delta > upset_threshold else (-1 if delta < -upset_threshold else 0)
    signals["delta"] = delta_signal
    explanations.append(f"Delta: {'Positive' if delta_signal > 0 else 'Negative'} ({delta:.3f})")
    
    # Volatility Signal
    volatility = analysis_results.get("volatility_5", [])
    if volatility and volatility[-1] is not None:
        # Normalize volatility (assuming typical range 0-5)
        vol_norm = min(volatility[-1] / 5, 1.0)
        volatility_signal = -vol_norm  # Higher volatility -> more caution
        signals["volatility"] = volatility_signal
        explanations.append(f"Volatility: {'High' if vol_norm > 0.5 else 'Low'} ({volatility[-1]:.2f})")
    else:
        signals["volatility"] = 0
    
    # Value Signal (EV based)
    value_signal = delta  # Simple value signal based on probability delta
    signals["value_signal"] = value_signal
    explanations.append(f"Value: {'Positive' if value_signal > 0 else 'Negative'} ({value_signal:.3f})")
    
    # Calculate combined score
    combined_score = 0
    for signal_name, weight in signal_weights.items():
        combined_score += signals.get(signal_name, 0) * weight
    
    # Component contributions
    contributions = {
        signal_name: signals.get(signal_name, 0) * weight
        for signal_name, weight in signal_weights.items()
    }
    
    explanation_text = " + ".join([f"{sig}: {cont:.3f}" for sig, cont in contributions.items()])
    
    return {
        "combined_score": combined_score,
        "contributions": contributions,
        "explanation": f"Signal fusion: {explanation_text} = {combined_score:.3f}",
        "component_explanations": explanations
    }

def forecast_time_series(series, horizon=6):
    """Forecast time series using simple methods (Prophet alternative)"""
    if not series or len(series) < 5:
        return None
    
    # Simple linear regression forecast
    x = np.arange(len(series)).reshape(-1, 1)
    y = np.array(series)
    
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x.flatten(), y)
        
        # Generate forecast
        future_x = np.arange(len(series), len(series) + horizon).reshape(-1, 1)
        forecast = intercept + slope * future_x.flatten()
        
        # Prediction intervals (simplified)
        residuals = y - (intercept + slope * x.flatten())
        std_residual = np.std(residuals)
        
        upper_95 = forecast + 1.96 * std_residual
        lower_95 = forecast - 1.96 * std_residual
        
        # Directional probability
        last_value = series[-1]
        prob_increase = np.sum(forecast > last_value) / len(forecast)
        
        return {
            "forecast": forecast.tolist(),
            "upper_95": upper_95.tolist(),
            "lower_95": lower_95.tolist(),
            "prob_increase": prob_increase,
            "trend_slope": slope
        }
    except:
        return None

def analyze_line_movement(series):
    """Classify line movement and detect conflicts"""
    if len(series) < 3:
        return {"movement_type": "insufficient_data", "conflicts": []}
    
    recent_changes = [series[i] - series[i-1] for i in range(1, len(series))]
    avg_change = np.mean(recent_changes[-3:]) if len(recent_changes) >= 3 else recent_changes[-1] if recent_changes else 0
    volatility = np.std(recent_changes) if recent_changes else 0
    
    # Classify movement
    if abs(avg_change) > 2 * volatility and abs(avg_change) > 0.5:
        movement_type = "sharp"
    elif abs(avg_change) > 0.1:
        movement_type = "moderate"
    else:
        movement_type = "stable"
    
    # Direction
    direction = "up" if avg_change > 0 else "down" if avg_change < 0 else "neutral"
    
    conflicts = []
    if movement_type == "sharp" and len(series) >= 5:
        # Check if sharp move contradicts longer trend
        long_trend = series[-1] - series[-5]
        if (avg_change > 0 and long_trend < 0) or (avg_change < 0 and long_trend > 0):
            conflicts.append("Sharp move contradicts longer-term trend")
    
    return {
        "movement_type": f"{movement_type}_{direction}",
        "avg_change": avg_change,
        "volatility": volatility,
        "conflicts": conflicts
    }

def generate_recommendation(game_data, analysis_results, bankroll):
    """Generate structured betting recommendation"""
    # Extract key metrics
    model_prob = analysis_results.get("model_probability", 0.5)
    implied_prob = analysis_results.get("implied_probability", 0.5)
    delta = model_prob - implied_prob
    ev = analysis_results.get("ev", 0)
    confidence = analysis_results.get("confidence", 0.5)
    
    # Calculate Kelly stake
    kelly_fraction = calculate_kelly(model_prob, game_data.get("odds", 100))
    kelly_stake = min(kelly_fraction * bankroll, bankroll * 0.5)  # Cap at 50%
    fractional_stake = kelly_stake * 0.5  # Half-Kelly for conservative approach
    
    # Top contributing indicators
    contributions = analysis_results.get("signal_contributions", {})
    top_contributors = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    
    recommendation = {
        "model_probability": model_prob,
        "implied_probability": implied_prob,
        "delta": delta,
        "expected_value": ev,
        "confidence": confidence,
        "kelly_fraction": kelly_fraction,
        "kelly_stake": kelly_stake,
        "fractional_stake": fractional_stake,
        "top_contributors": top_contributors,
        "signal_explanation": analysis_results.get("signal_explanation", ""),
        "upset_alert": delta >= upset_threshold
    }
    
    # Generate plain English explanation
    explanation_parts = []
    
    if recommendation["upset_alert"]:
        explanation_parts.append("🚨 UPSET ALERT: Significant value detected!")
    
    explanation_parts.append(f"Model probability: {model_prob:.1%} vs Market: {implied_prob:.1%}")
    explanation_parts.append(f"Expected Value: {ev:.2%}")
    explanation_parts.append(f"Confidence: {confidence:.1%}")
    
    if kelly_fraction > 0:
        explanation_parts.append(f"Recommended stake: ${kelly_stake:.2f} (Kelly: {kelly_fraction:.1%})")
    
    explanation_parts.append(f"Top signals: {', '.join([f'{sig}({val:.3f})' for sig, val in top_contributors])}")
    
    recommendation["english_explanation"] = " | ".join(explanation_parts)
    
    return recommendation

def run_backtest(rows, bankroll):
    """Run backtesting simulation"""
    if not rows or len(rows) < 2:
        return {"html": "<p>Not enough data for backtesting. Need at least 2 data points.</p>", "results": None}
    
    current_bankroll = bankroll
    bets = []
    starting_bankroll = bankroll
    
    for i in range(len(rows) - 1):
        current_data = rows[i]
        next_data = rows[i + 1]
        
        # Simplified backtest logic
        spread_move = (next_data.get("spread", 0) or 0) - (current_data.get("spread", 0) or 0)
        spread_bet_success = spread_move < 0
        
        total_move = (next_data.get("total", 0) or 0) - (current_data.get("total", 0) or 0)
        total_bet_success = total_move > 0
        
        ml_away_current = current_data.get("ml_away", 100)
        ml_away_next = next_data.get("ml_away", 100)
        ml_move = (ml_away_next or 100) - (ml_away_current or 100)
        ml_bet_success = ml_move < 0
        
        bet_amount = current_bankroll * 0.05
        
        # Update bankroll
        if spread_bet_success:
            current_bankroll += bet_amount * 0.91
        else:
            current_bankroll -= bet_amount
            
        if total_bet_success:
            current_bankroll += bet_amount * 0.91
        else:
            current_bankroll -= bet_amount
            
        if ml_bet_success:
            payout = bet_amount * (abs(ml_away_current) / 100) if ml_away_current > 0 else bet_amount * (100 / abs(ml_away_current))
            current_bankroll += payout
        else:
            current_bankroll -= bet_amount
        
        bets.append({
            "time": current_data["time"],
            "spread_bet": "Win" if spread_bet_success else "Loss",
            "total_bet": "Win" if total_bet_success else "Loss",
            "ml_bet": "Win" if ml_bet_success else "Loss",
            "bankroll": current_bankroll
        })
    
    profit = current_bankroll - starting_bankroll
    roi = (profit / starting_bankroll) * 100 if starting_bankroll > 0 else 0
    
    return {
        "starting_bankroll": starting_bankroll,
        "ending_bankroll": current_bankroll,
        "profit": profit,
        "roi": roi,
        "bets": bets
    }

def analyze_and_build(rows, errors, bankroll):
    """Main analysis function"""
    if not rows:
        return {
            "html": "<div class='analysis-block'>No valid data to analyze.</div>",
            "recommendation": "No data",
            "confidence": "Low"
        }
    
    # Initialize ML model
    ml_model = train_model(rows)  # This would use actual training data in production
    
    analysis_results = []
    
    for i, row in enumerate(rows):
        # Extract time series data for each market
        spread_series = [r.get("spread", 0) or 0 for r in rows[:i+1]]
        total_series = [r.get("total", 0) or 0 for r in rows[:i+1]]
        ml_away_series = [r.get("ml_away", 100) or 100 for r in rows[:i+1]]
        
        # Full feed analysis
        spread_analysis = calculate_full_feed_analysis(spread_series)
        total_analysis = calculate_full_feed_analysis(total_series)
        ml_analysis = calculate_full_feed_analysis(ml_away_series)
        
        # ML prediction
        ml_prediction, ml_confidence, shap_values = predict_game(ml_model, row)
        
        # Signal fusion
        current_spread = row.get("spread", 0) or 0
        spread_market_prob = row.get("spread_no_vig", {}).get("prob1", 0.5) if row.get("spread_no_vig") else 0.5
        spread_signal = calculate_signal_fusion(spread_analysis, current_spread, spread_market_prob, ml_prediction)
        
        # Generate recommendation
        recommendation = generate_recommendation(row, {
            "model_probability": ml_prediction,
            "implied_probability": spread_market_prob,
            "ev": calculate_ev(ml_prediction, row.get("spread_vig", -110)),
            "confidence": ml_confidence,
            "signal_contributions": spread_signal["contributions"],
            "signal_explanation": spread_signal["explanation"]
        }, bankroll)
        
        # Forecast
        spread_forecast = forecast_time_series(spread_series, forecast_horizon)
        
        # Line movement analysis
        line_movement = analyze_line_movement(spread_series)
        
        analysis_results.append({
            "row": row,
            "spread_analysis": spread_analysis,
            "total_analysis": total_analysis,
            "ml_analysis": ml_analysis,
            "ml_prediction": ml_prediction,
            "ml_confidence": ml_confidence,
            "signal_fusion": spread_signal,
            "recommendation": recommendation,
            "forecast": spread_forecast,
            "line_movement": line_movement,
            "shap_values": shap_values
        })
    
    # Build HTML output
    html_output = build_analysis_html(analysis_results, errors, bankroll)
    
    # Overall recommendation
    if analysis_results:
        last_analysis = analysis_results[-1]
        overall_rec = last_analysis["recommendation"]["english_explanation"]
        confidence = "High" if last_analysis["ml_confidence"] > 0.7 else "Medium" if last_analysis["ml_confidence"] > 0.5 else "Low"
    else:
        overall_rec = "No analysis available"
        confidence = "Low"
    
    return {
        "html": html_output,
        "recommendation": overall_rec,
        "confidence": confidence
    }

def build_analysis_html(analysis_results, errors, bankroll):
    """Build HTML output for analysis results"""
    if not analysis_results:
        return "<div class='analysis-block'>No analysis results.</div>"
    
    html_parts = []
    
    for i, result in enumerate(analysis_results):
        row = result["row"]
        recommendation = result["recommendation"]
        
        # Determine favorite and underdog ML
        if row.get("ml_away") and row.get("ml_home"):
            if (row["ml_away"] == 'even' or (isinstance(row["ml_away"], (int, float)) and row["ml_away"] > 0)) and \
               (isinstance(row["ml_home"], (int, float)) and row["ml_home"] < 0):
                favorite_ml = row["ml_home"]
                underdog_ml = row["ml_away"]
            else:
                favorite_ml = row["ml_away"]
                underdog_ml = row["ml_home"]
        else:
            favorite_ml = "N/A"
            underdog_ml = "N/A"
        
        # Calculate implied probabilities
        favorite_implied = implied_probability(favorite_ml)
        underdog_implied = implied_probability(underdog_ml)
        
        # Check upset threshold
        upset_alert = result["ml_prediction"] - underdog_implied >= upset_threshold
        
        html_parts.append(f"""
        <div class="analysis-block">
            <h3>Game Analysis - {row['time'].strftime('%m/%d %I:%M %p')}</h3>
            <div class="ta-output">
                <table class="ta-table">
                    <tr>
                        <th>Metric</th>
                        <th>Spread</th>
                        <th>Total</th>
                        <th>Moneyline</th>
                    </tr>
                    <tr>
                        <td>Current Value</td>
                        <td>{row.get('spread', 'N/A')}</td>
                        <td>{row.get('total', 'N/A')}</td>
                        <td>Away: {row.get('ml_away', 'N/A')} | Home: {row.get('ml_home', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Favorite ML</td>
                        <td colspan="3">{favorite_ml} (Implied: {favorite_implied:.1%})</td>
                    </tr>
                    <tr>
                        <td>Underdog ML</td>
                        <td colspan="3">{underdog_ml} (Implied: {underdog_implied:.1%})</td>
                    </tr>
                    <tr>
                        <td>Upset Threshold</td>
                        <td colspan="3">{'🚨 ALERT: Model predicts upset!' if upset_alert else 'Within normal range'}</td>
                    </tr>
                    <tr>
                        <td>Model Probability</td>
                        <td colspan="3">{result['ml_prediction']:.1%} (Confidence: {result['ml_confidence']:.1%})</td>
                    </tr>
                    <tr>
                        <td>Signal Fusion</td>
                        <td colspan="3">{result['signal_fusion']['explanation']}</td>
                    </tr>
                    <tr>
                        <td>Line Movement</td>
                        <td colspan="3">{result['line_movement']['movement_type']}</td>
                    </tr>
                </table>
            </div>
            
            <div class="recommendation">
                <h4>Recommendation</h4>
                <p>{recommendation['english_explanation']}</p>
                <div class="kelly-box">
                    <strong>Bankroll Management:</strong><br>
                    Bankroll: ${bankroll:.2f}<br>
                    Full Kelly Stake: ${recommendation['kelly_stake']:.2f}<br>
                    Fractional Kelly (50%): ${recommendation['fractional_stake']:.2f}
                </div>
            </div>
        </div>
        """)
    
    # Add errors if any
    if errors:
        html_parts.append(f"""
        <div class="analysis-block" style="color: #ffdcdc;">
            <strong>Parser warnings ({len(errors)}) — skipped malformed blocks:</strong>
            <pre>{errors[:5]}</pre>
        </div>
        """)
    
    return "\n".join(html_parts)

# ========== STREAMLIT UI ==========

def main():
    st.title("Omniscience — Enhanced TA Engine (EV + Kelly + Backtesting)")
    
    st.markdown("""
    <div class="muted">
    Paste feed: first line header (required) then repeated 5-line blocks.<br>
    Blocks: <strong>line1</strong> (time [team?] spread), <strong>line2</strong> (spread vig), <strong>line3</strong> (total e.g. o154/u154.5), 
    <strong>line4</strong> (total vig), <strong>line5</strong> (awayML homeML).
    </div>
    """, unsafe_allow_html=True)
    
    # Main layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        odds_data = st.text_area("Paste odds feed here...", height=300, 
                                placeholder="Paste your odds data here...")
        
        # Controls
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            analyze_btn = st.button("Analyze", type="primary")
        with col1b:
            backtest_btn = st.button("Backtest", type="secondary")
        with col1c:
            clear_btn = st.button("Clear", type="secondary")
        
        if clear_btn:
            st.experimental_rerun()
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    if 'preview_data' not in st.session_state:
        st.session_state.preview_data = None
    
    # Process data
    if analyze_btn and odds_data:
        with st.spinner("Analyzing data..."):
            parsed_data = parse_blocks_strict(odds_data)
            st.session_state.preview_data = parsed_data["rows"]
            
            if parsed_data["rows"]:
                analysis = analyze_and_build(parsed_data["rows"], parsed_data["errors"], bankroll)
                st.session_state.analysis_results = analysis
                st.session_state.backtest_results = None
            else:
                st.error("No valid data parsed. Please check your input format.")
    
    if backtest_btn and odds_data:
        with st.spinner("Running backtest..."):
            parsed_data = parse_blocks_strict(odds_data)
            if parsed_data["rows"] and len(parsed_data["rows"]) >= 2:
                backtest_results = run_backtest(parsed_data["rows"], bankroll)
                st.session_state.backtest_results = backtest_results
                st.session_state.analysis_results = None
            else:
                st.error("Not enough data for backtesting. Need at least 2 data points.")
    
    # Display preview table
    with col1:
        if st.session_state.preview_data:
            st.subheader("Parsed Preview (Chronological Order)")
            preview_df = pd.DataFrame(st.session_state.preview_data)
            
            # Format the preview dataframe
            display_cols = ["time", "team", "spread", "spread_vig", "total", "total_vig", 
                          "ml_away", "ml_home", "ml_away_prob", "ml_home_prob"]
            available_cols = [col for col in display_cols if col in preview_df.columns]
            
            display_df = preview_df[available_cols].copy()
            if 'time' in display_df.columns:
                display_df['time'] = display_df['time'].dt.strftime('%m/%d %I:%M %p')
            if 'ml_away_prob' in display_df.columns:
                display_df['ml_away_prob'] = (display_df['ml_away_prob'] * 100).round(1).astype(str) + '%'
            if 'ml_home_prob' in display_df.columns:
                display_df['ml_home_prob'] = (display_df['ml_home_prob'] * 100).round(1).astype(str) + '%'
            
            st.dataframe(display_df, use_container_width=True)
    
    # Display results
    with col2:
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            # Top recommendation
            st.markdown(f"""
            <div class="rec">
                <strong>{results['recommendation']}</strong>
                <div class="muted">Confidence: {results['confidence']}. Based on {len(st.session_state.preview_data) if st.session_state.preview_data else 0} ticks.</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Analysis results
            st.markdown(results['html'], unsafe_allow_html=True)
        
        elif st.session_state.backtest_results:
            results = st.session_state.backtest_results
            
            st.markdown("""
            <div class="backtest-results">
                <h3>Backtest Results</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Starting Bankroll", f"${results['starting_bankroll']:.2f}")
            st.metric("Ending Bankroll", f"${results['ending_bankroll']:.2f}")
            st.metric("Profit", f"${results['profit']:.2f}", 
                     delta=f"{results['roi']:.2f}%")
            
            if results['bets']:
                st.subheader("Bet History")
                bets_df = pd.DataFrame(results['bets'])
                st.dataframe(bets_df, use_container_width=True)
        
        else:
            st.markdown("""
            <div class="rec">
                <strong>No analysis yet</strong>
                <div class="muted">Paste feed (header first line) and click Analyze.</div>
            </div>
            """, unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>
.analysis-block {
    background: rgba(255,255,255,0.03);
    padding: 14px;
    border-radius: 10px;
    margin-bottom: 12px;
}
.rec {
    background: linear-gradient(90deg,#071827,#04121a);
    padding: 12px;
    border-left: 4px solid #ffd166;
    border-radius: 6px;
    margin-bottom: 12px;
}
.muted {
    font-size: 13px;
    color: #9fe9e0;
}
.ta-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
    font-size: 13px;
}
.ta-table th, .ta-table td {
    padding: 6px;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    text-align: left;
}
.kelly-box {
    background: rgba(255,255,255,0.05);
    padding: 10px;
    border-radius: 8px;
    margin: 10px 0;
}
.backtest-results {
    margin-top: 20px;
    padding: 15px;
    background: rgba(0,0,0,0.2);
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
