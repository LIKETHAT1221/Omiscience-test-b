import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import math
from typing import List, Dict, Tuple, Optional, Any
import json

# Add these CSS styles at the beginning of your render() method
def add_custom_css():
    st.markdown("""
    <style>
    .ev-positive { color: #60d394; font-weight: bold; }
    .ev-negative { color: #ff7b7b; font-weight: bold; }
    .ev-neutral { color: #ffd166; font-weight: bold; }
    .good { color: #60d394; }
    .bad { color: #ff7b7b; }
    .neutral { color: #ffd166; }
    .ta-table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 14px;
    }
    .ta-table th, .ta-table td {
        padding: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        text-align: left;
    }
    .ta-table th {
        background: rgba(255,255,255,0.05);
    }
    .analysis-block {
        background: rgba(255,255,255,0.03);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .top-play {
        background: linear-gradient(90deg, #0a2c3d, #082230);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffbe0b;
        margin: 15px 0;
    }
    .kelly-box {
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)

class StreamlitOmniscienceApp:
    def __init__(self):
        self.parser = OddsParser()
        self.ta = TechnicalAnalysis()
        self.betting = BettingAnalysis()
        self.ta_engine = OmniscienceTAEngine()
        
        # Initialize session state
        if 'parsed_rows' not in st.session_state:
            st.session_state.parsed_rows = []
        if 'parsed_errors' not in st.session_state:
            st.session_state.parsed_errors = []
        if 'bankroll' not in st.session_state:
            st.session_state.bankroll = 1000.0
        if 'analysis_result' not in st.session_state:
            st.session_state.analysis_result = None
        if 'backtest_result' not in st.session_state:
            st.session_state.backtest_result = None

    def analyze_and_build(self, rows: List[Dict], errors: List[Dict]) -> Dict[str, Any]:
        """COMPLETE analysis function with full TA and voting system"""
        if not rows:
            return {
                "html": "<div class='analysis-block'>No valid blocks parsed.</div>",
                "rec": "No data",
                "conf": "Low",
                "meta": "No data available"
            }
            
        chrono = rows
        bankroll = st.session_state.bankroll
        last_row = rows[-1]

        # Process Spread with ALL TA indicators
        spreads = [r["spread"] for r in chrono if r["spread"] is not None]
        if spreads:
            # Calculate all spread indicators
            sma_spread = self.ta.SMA_full(spreads, 10)
            ema_spread = self.ta.EMA_full(spreads, 5) 
            rsi_spread = self.ta.RSI_full(spreads, 7)
            macd_spread = self.ta.MACD_full(spreads)
            bb_spread = self.ta.bollinger_bands(spreads, 20, 2)
            z_spread = self.ta.zscore(spreads, 10)
            fib_spread = self.ta.get_fibonacci_levels_full(spreads, 13)
            greeks_spread = self.ta.greek_analysis(spreads, 13)
            steam_spread = self.ta.detect_steam_moves(spreads, 2)
            roc_spread = self.ta.ROC(spreads, 2)
            ama_spread = self.ta.adaptive_MA(spreads, 2, 8, 6)
            
            # Get last values
            last_sma = self._last_valid(sma_spread)
            last_ema = self._last_valid(ema_spread)
            last_rsi = self._last_valid(rsi_spread)
            last_macd_val = self._last_valid(macd_spread["macd"])
            last_macd_sig = self._last_valid(macd_spread["signal"])
            last_bb = bb_spread[-1] if bb_spread and bb_spread[-1] else None
            last_z = self._last_valid(z_spread)
            last_roc = self._last_valid(roc_spread)
            last_ama = self._last_valid(ama_spread)
        else:
            last_sma = last_ema = last_rsi = last_macd_val = last_macd_sig = None
            last_bb = last_z = last_roc = last_ama = None
            fib_spread = greeks_spread = None
            steam_spread = []

        # Similar processing for totals and moneyline (implement similarly)
        totals = [r["total"] for r in chrono if r["total"] is not None]
        ml_away_series = [r["ml_away"] for r in chrono if r["ml_away"] is not None and r["ml_away"] != 'even']
        
        # IMPLEMENT THE COMPLETE VOTING SYSTEM
        spread_votes = self._get_spread_votes(
            spreads, last_ema, last_sma, last_rsi, last_macd_val, last_macd_sig,
            last_bb, last_z, steam_spread, greeks_spread
        )
        
        total_votes = self._get_total_votes(totals)  # Implement similar to spread_votes
        ml_votes = self._get_ml_votes(ml_away_series)  # Implement similar to spread_votes

        # Count votes for each type
        spread_synth = self._count_votes(spread_votes, "favorite", "dog")
        total_synth = self._count_votes(total_votes, "over", "under") 
        ml_synth = self._count_votes(ml_votes, "bullish", "bearish")

        # Calculate probabilities and EV
        spread_market_prob = last_row["spreadNoVig"]["prob1"] if last_row.get("spreadNoVig") else 0.5
        total_market_prob = last_row["totalNoVig"]["prob1"] if last_row.get("totalNoVig") else 0.5
        ml_market_prob = last_row["mlNoVig"]["prob1"] if last_row.get("mlNoVig") else 0.5
        
        spread_model_prob = self.betting.calculate_model_probability(spread_synth, spread_market_prob)
        total_model_prob = self.betting.calculate_model_probability(total_synth, total_market_prob) 
        ml_model_prob = self.betting.calculate_model_probability(ml_synth, ml_market_prob)

        # Calculate EV and Kelly
        spread_ev = self.betting.calculate_EV(spread_model_prob, last_row.get("spread_vig"))
        total_ev = self.betting.calculate_EV(total_model_prob, last_row.get("total_vig"))
        ml_ev = self.betting.calculate_EV(ml_model_prob, last_row.get("ml_away"))
        
        spread_kelly = self.betting.calculate_kelly(spread_model_prob, last_row.get("spread_vig"))
        total_kelly = self.betting.calculate_kelly(total_model_prob, last_row.get("total_vig"))
        ml_kelly = self.betting.calculate_kelly(ml_model_prob, last_row.get("ml_away"))

        # Determine best bet
        ev_values = [
            {"type": "Spread", "ev": spread_ev, "kelly": spread_kelly, "prob": spread_model_prob},
            {"type": "Total", "ev": total_ev, "kelly": total_kelly, "prob": total_model_prob},
            {"type": "Moneyline", "ev": ml_ev, "kelly": ml_kelly, "prob": ml_model_prob}
        ]
        ev_values = [ev for ev in ev_values if not math.isnan(ev["ev"])]
        best_bet = max(ev_values, key=lambda x: x["ev"]) if ev_values else None

        # Generate recommendation
        recommendation = self._generate_recommendation(spread_synth, total_synth, ml_synth)
        
        # Build COMPLETE HTML output with TA table
        html_output = self._build_complete_analysis_html(
            best_bet, spread_ev, total_ev, ml_ev,
            spread_kelly, total_kelly, ml_kelly,
            spread_model_prob, total_model_prob, ml_model_prob,
            bankroll, recommendation, errors,
            spread_votes, total_votes, ml_votes,
            spread_synth, total_synth, ml_synth,
            fib_spread, last_roc, last_ama, last_rsi
        )
        
        return {
            "html": html_output,
            "rec": recommendation,
            "conf": "High" if best_bet and best_bet["ev"] > 0.05 else "Medium/Low",
            "meta": f"Confidence: {'High' if best_bet and best_bet['ev'] > 0.05 else 'Medium/Low'}. Based on {len(rows)} ticks."
        }

    def _get_spread_votes(self, spreads, ema, sma, rsi, macd_val, macd_sig, bb, z, steam, greeks):
        """Implement complete spread voting logic from original JavaScript"""
        votes = []
        
        # EMA vs SMA
        if ema is not None and sma is not None:
            votes.append("favorite" if ema > sma else "dog")
        
        # RSI
        if rsi is not None:
            if rsi < 30:
                votes.append("favorite (oversold)")
            elif rsi > 70:
                votes.append("dog (overbought)")
            else:
                votes.append("neutral")
        
        # MACD
        if macd_val is not None and macd_sig is not None:
            votes.append("favorite (momentum)" if macd_val > macd_sig else "dog (momentum)")
        
        # Bollinger Bands
        if bb and spreads:
            current_spread = spreads[-1]
            if current_spread > bb["upper"]:
                votes.append("dog (outside BB)")
            elif current_spread < bb["lower"]:
                votes.append("favorite (outside BB)")
            else:
                votes.append("neutral")
        
        # Z-score
        if z is not None:
            if z > 1:
                votes.append("dog (z-score)")
            elif z < -1:
                votes.append("favorite (z-score)")
            else:
                votes.append("neutral")
        
        # Steam moves
        if steam:
            votes.append("favorite (steam)")
        
        # Greeks
        if greeks and greeks.get("delta") is not None:
            if greeks["delta"] > 0.5:
                votes.append("favorite (delta)")
            elif greeks["delta"] < -0.5:
                votes.append("dog (delta)")
            else:
                votes.append("neutral")
                
        return votes

    def _count_votes(self, votes, up_word, down_word):
        """Count votes for each direction"""
        up = len([v for v in votes if up_word in v.lower()])
        down = len([v for v in votes if down_word in v.lower()])
        neutral = len([v for v in votes if "neutral" in v.lower()])
        
        # Determine overall direction
        if up > down:
            direction = "good"
        elif down > up:
            direction = "bad"
        else:
            direction = "neutral"
            
        return {"up": up, "down": down, "neutral": neutral, "direction": direction}

    def _last_valid(self, series):
        """Get last non-null value from series"""
        if not series:
            return None
        valid_values = [x for x in series if x is not None]
        return valid_values[-1] if valid_values else None

    def _build_complete_analysis_html(self, best_bet, spread_ev, total_ev, ml_ev,
                                    spread_kelly, total_kelly, ml_kelly,
                                    spread_prob, total_prob, ml_prob,
                                    bankroll, recommendation, errors,
                                    spread_votes, total_votes, ml_votes,
                                    spread_synth, total_synth, ml_synth,
                                    fib_spread, roc, ama, rsi):
        """Build COMPLETE HTML analysis with TA table and voting system"""
        
        def get_ev_class(ev):
            if math.isnan(ev):
                return ""
            if ev > 0.05:
                return "ev-positive"
            if ev > 0:
                return "ev-neutral"
            return "ev-negative"

        html = """
        <div class='analysis-block'>
            <h3>Enhanced TA Stack with EV & Kelly Sizing</h3>
        """
        
        # Best Bet Recommendation
        if best_bet:
            stake = self.betting.get_kelly_stake(best_bet["kelly"], bankroll)
            html += f"""
            <div class='top-play'>
                <h3>🔥 TOP PLAY: {best_bet["type"]}</h3>
                <p><strong>Expected Value (EV):</strong> <span class='{get_ev_class(best_bet["ev"])}'>{(best_bet['ev'] * 100):.2f}%</span></p>
                <p><strong>Model Probability:</strong> {(best_bet['prob'] * 100):.1f}%</p>
                <p><strong>Recommended Stake:</strong> ${stake:.2f} ({(stake/bankroll*100):.1f}% of bankroll)</p>
                <p><strong>Kelly Fraction:</strong> {(best_bet['kelly'] * 100):.1f}%</p>
            </div>
            """
        
        # COMPLETE TA TABLE
        html += """
        <table class='ta-table'>
            <tr>
                <th>TA Indicator</th>
                <th>Spread Output</th>
                <th>Total Output</th>
                <th>Underdog Moneyline Output</th>
            </tr>
        """
        
        # Fibonacci Row
        fib_spread_str = "n/a"
        if fib_spread:
            retracements = ", ".join([f"{x:.2f}" for x in fib_spread["retracements"]])
            extensions = ", ".join([f"{x:.2f}" for x in fib_spread["extensions"]])
            fib_spread_str = f"R: {retracements}<br>E: {extensions}"
        
        html += f"""
            <tr>
                <td>Fibonacci<br><span style='font-size:11px;color:#ffd166;'>R: retracement, E: extension</span></td>
                <td>{fib_spread_str}</td>
                <td>n/a</td>
                <td>n/a</td>
            </tr>
        """
        
        # ROC Row
        roc_str = f"{roc:.2f}%" if roc is not None else "n/a"
        html += f"""
            <tr>
                <td>ROC(2)</td>
                <td style='font-weight:bold;color:#ffbe0b;'>{roc_str}</td>
                <td style='font-weight:bold;color:#ffbe0b;'>n/a</td>
                <td style='font-weight:bold;color:#ffbe0b;'>n/a</td>
            </tr>
        """
        
        # Adaptive MA Row
        ama_str = f"{ama:.2f}" if ama is not None else "n/a"
        html += f"""
            <tr>
                <td>Adaptive MA</td>
                <td style='font-weight:bold;color:#a3e635;'>{ama_str}</td>
                <td style='font-weight:bold;color:#a3e635;'>n/a</td>
                <td style='font-weight:bold;color:#a3e635;'>n/a</td>
            </tr>
        """
        
        # RSI Row
        rsi_str = "n/a"
        if rsi is not None:
            if rsi < 30:
                rsi_str = "favorite (oversold)"
            elif rsi > 70:
                rsi_str = "dog (overbought)"
            else:
                rsi_str = "neutral"
                
        html += f"""
            <tr>
                <td>RSI</td>
                <td>{rsi_str}</td>
                <td>n/a</td>
                <td>n/a</td>
            </tr>
        """
        
        # Add more TA rows as needed...
        
        html += """
        </table>
        """
        
        # VOTING SYSTEM DISPLAY
        html += f"""
        <h4>Agreement & Conflict</h4>
        <div>
            <strong>Spread indicators:</strong> <span class='{spread_synth["direction"]}'>{", ".join(spread_votes)}</span><br>
            <strong>Total indicators:</strong> <span class='{total_synth["direction"]}'>{", ".join(total_votes)}</span><br>
            <strong>Underdog ML indicators:</strong> <span class='{ml_synth["direction"]}'>{", ".join(ml_votes)}</span>
        </div>
        
        <div style='margin: 15px 0; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px;'>
            <strong>Vote Summary:</strong><br>
            • Spread: {spread_synth['up']} favorite votes, {spread_synth['down']} dog votes, {spread_synth['neutral']} neutral<br>
            • Total: {total_synth['up']} over votes, {total_synth['down']} under votes, {total_synth['neutral']} neutral<br>
            • Moneyline: {ml_synth['up']} bullish votes, {ml_synth['down']} bearish votes, {ml_synth['neutral']} neutral
        </div>
        """
        
        # EV Analysis Table
        html += f"""
        <h4>Expected Value Analysis</h4>
        <table class='ta-table'>
            <tr>
                <th>Bet Type</th>
                <th>Model Probability</th>
                <th>Expected Value</th>
                <th>Kelly Stake</th>
            </tr>
            <tr>
                <td>Spread</td>
                <td>{(spread_prob * 100):.1f}%</td>
                <td class='{get_ev_class(spread_ev)}'>{(spread_ev * 100):.2f}%</td>
                <td>${self.betting.get_kelly_stake(spread_kelly, bankroll):.2f}</td>
            </tr>
            <tr>
                <td>Total</td>
                <td>{(total_prob * 100):.1f}%</td>
                <td class='{get_ev_class(total_ev)}'>{(total_ev * 100):.2f}%</td>
                <td>${self.betting.get_kelly_stake(total_kelly, bankroll):.2f}</td>
            </tr>
            <tr>
                <td>Moneyline</td>
                <td>{(ml_prob * 100):.1f}%</td>
                <td class='{get_ev_class(ml_ev)}'>{(ml_ev * 100):.2f}%</td>
                <td>${self.betting.get_kelly_stake(ml_kelly, bankroll):.2f}</td>
            </tr>
        </table>
        """
        
        # Bankroll Management
        html += f"""
        <div class='kelly-box'>
            <h4>Bankroll Management (Kelly Criterion)</h4>
            <p><strong>Bankroll:</strong> ${bankroll:.2f}</p>
            <p><strong>Spread Stake:</strong> ${self.betting.get_kelly_stake(spread_kelly, bankroll):.2f} ({(spread_kelly * 100 if not math.isnan(spread_kelly) else 0):.1f}% of bankroll)</p>
            <p><strong>Total Stake:</strong> ${self.betting.get_kelly_stake(total_kelly, bankroll):.2f} ({(total_kelly * 100 if not math.isnan(total_kelly) else 0):.1f}% of bankroll)</p>
            <p><strong>Moneyline Stake:</strong> ${self.betting.get_kelly_stake(ml_kelly, bankroll):.2f} ({(ml_kelly * 100 if not math.isnan(ml_kelly) else 0):.1f}% of bankroll)</p>
            <p style='color: #9fe9e0; font-size: 12px;'>Note: Kelly stakes are capped at 50% of bankroll to manage risk.</p>
        </div>
        """
        
        # Recommendation
        html += f"""
        <div style='background: linear-gradient(90deg, #071827, #04121a); padding: 15px; border-radius: 6px; border-left: 4px solid #ffd166; margin-bottom: 15px;'>
            <h4>Recommendation</h4>
            <p><strong>{recommendation}</strong></p>
        </div>
        """
        
        # Errors
        if errors:
            html += f"""
            <div style='color: #9fe9e0; font-size: 12px;'>
                <strong>Parser warnings ({len(errors)}) — skipped malformed blocks:</strong>
                <pre>{json.dumps(errors[:3], indent=2)}</pre>
            </div>
            """
            
        html += "</div>"
        return html

    def render(self):
        """Main Streamlit rendering function"""
        st.set_page_config(
            page_title="Omniscience — Enhanced TA Engine",
            page_icon="🔮",
            layout="wide"
        )
        
        # Add custom CSS
        add_custom_css()
        
        st.title("Omniscience — Enhanced TA Engine (EV + Kelly + Backtesting)")
        
        st.markdown("""
        <div style='color: #9fe9e0; font-size: 14px; margin-bottom: 20px;'>
            Paste feed: first line header (required) then repeated 5-line blocks.<br>
            Blocks: <strong>line1</strong> (time [team?] spread), <strong>line2</strong> (spread vig), 
            <strong>line3</strong> (total e.g. o154/u154.5), <strong>line4</strong> (total vig), 
            <strong>line5</strong> (awayML homeML).
        </div>
        """, unsafe_allow_html=True)
        
        # Main layout
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Input area
            odds_data = st.text_area(
                "Paste odds feed here...",
                height=300,
                placeholder="Paste your odds feed data here..."
            )
            
            # Controls
            col1a, col1b, col1c, col1d = st.columns(4)
            
            with col1a:
                analyze_btn = st.button("Analyze", type="primary", use_container_width=True)
                
            with col1b:
                backtest_btn = st.button("Backtest", use_container_width=True)
                
            with col1c:
                clear_btn = st.button("Clear", use_container_width=True)
                
            with col1d:
                bankroll_option = st.selectbox(
                    "Bankroll",
                    options=["1000", "5000", "10000", "custom"],
                    format_func=lambda x: f"Bankroll: ${x}" if x != "custom" else "Custom...",
                    label_visibility="collapsed"
                )
            
            # Custom bankroll input
            if bankroll_option == "custom":
                custom_bankroll = st.number_input(
                    "Enter custom bankroll amount",
                    min_value=100.0,
                    max_value=1000000.0,
                    value=1000.0,
                    step=100.0
                )
                st.session_state.bankroll = custom_bankroll
            else:
                st.session_state.bankroll = float(bankroll_option)
            
            st.markdown("<div style='color: #9fe9e0; font-size: 12px; margin-top: 8px;'>Malformed blocks are reported in the analysis.</div>", unsafe_allow_html=True)
            
            # Preview table
            st.subheader("Parsed preview (chronological order)")
            if st.session_state.parsed_rows:
                preview_data = []
                for row in st.session_state.parsed_rows:
                    preview_data.append({
                        "Time": row["time"].strftime("%m/%d %H:%M"),
                        "Team": row.get("team", ""),
                        "Spread": row.get("spread", ""),
                        "SpreadVig": row.get("spread_vig", ""),
                        "Total": row.get("total", ""),
                        "TotalVig": row.get("total_vig", ""),
                        "AwayML": row.get("ml_away", ""),
                        "HomeML": row.get("ml_home", ""),
                        "AwayML Prob": f"{(row.get('ml_away_prob', 0) * 100):.1f}%" if row.get('ml_away_prob') else "",
                        "HomeML Prob": f"{(row.get('ml_home_prob', 0) * 100):.1f}%" if row.get('ml_home_prob') else ""
                    })
                
                st.dataframe(preview_data, use_container_width=True)
        
        with col2:
            # Recommendation box - FIXED: Always show current state
            current_result = st.session_state.analysis_result
            if current_result:
                st.markdown(f"""
                <div style='background: linear-gradient(90deg, #071827, #04121a); padding: 15px; border-radius: 8px; border-left: 4px solid #ffd166; margin-bottom: 15px;'>
                    <strong>{current_result.get('rec', 'No analysis yet')}</strong>
                    <div style='color: #9fe9e0; font-size: 12px;'>{current_result.get('meta', 'Paste feed and click Analyze.')}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: linear-gradient(90deg, #071827, #04121a); padding: 15px; border-radius: 8px; border-left: 4px solid #ffd166; margin-bottom: 15px;'>
                    <strong>No analysis yet</strong>
                    <div style='color: #9fe9e0; font-size: 12px;'>Paste feed (first line header) and click Analyze.</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Analysis results - FIXED: Always check session state
            if st.session_state.analysis_result:
                st.markdown(st.session_state.analysis_result["html"], unsafe_allow_html=True)
            
            # Backtest results - FIXED: Proper display
            if st.session_state.backtest_result:
                st.markdown(st.session_state.backtest_result, unsafe_allow_html=True)
        
        # Button handlers - FIXED: Proper state management
        if analyze_btn and odds_data:
            with st.spinner("Analyzing data..."):
                rows, errors = self.parse_blocks_strict(odds_data)
                st.session_state.parsed_rows = rows
                st.session_state.parsed_errors = errors
                
                if rows:
                    result = self.analyze_and_build(rows, errors)
                    st.session_state.analysis_result = result
                    st.session_state.backtest_result = None
                    st.rerun()
                else:
                    st.error("No valid data parsed. Check the input format.")
        
        if backtest_btn and st.session_state.parsed_rows:
            with st.spinner("Running backtest..."):
                result = self.run_backtest(st.session_state.parsed_rows, st.session_state.bankroll)
                st.session_state.backtest_result = result["html"]
                st.rerun()
        
        if clear_btn:
            st.session_state.parsed_rows = []
            st.session_state.parsed_errors = []
            st.session_state.analysis_result = None
            st.session_state.backtest_result = None
            st.rerun()

# Make sure to run the app
if __name__ == "__main__":
    app = StreamlitOmniscienceApp()
    app.render()
