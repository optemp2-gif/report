import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import calendar
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MT5 Trade Analyzer", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
    /* --- FONTS & VARS --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-color: #0e1117;
        --card-bg: #1A1C24; /* Slightly lighter/blue-ish dark */
        --card-border: 1px solid rgba(255, 255, 255, 0.08);
        --text-primary: #E0E0E0;
        --text-secondary: #A0A0A0;
        --accent-success: #00D26A; /* Vibrant Green */
        --accent-danger: #F8312F; /* Vibrant Red */
        --accent-neutral: #888888;
        --accent-brand: #4E86FF; /* Blue brand color */
        --radius: 10px;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.15);
    }

    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Fix for Material Icons rendering as text */
    .material-icons, .icon-button, [data-testid="stExpander"] span[aria-hidden="true"] { 
        font-family: 'Material Icons' !important; 
    }

    /* --- GLOBAL STREAMLIT TWEAKS --- */
    #MainMenu, header, footer {
        visibility: hidden;
    }
    [data-testid="stSidebarCollapseButton"] {
        display: none;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100% !important;
    }
    
    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {
        background-color: #12141C;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    [data-testid="stSidebar"] h3 {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 20px;
    }
    
    /* --- CUSTOM CARDS --- */
    .metric-card {
        background-color: var(--card-bg);
        border: var(--card-border);
        border-radius: var(--radius);
        padding: 15px; /* Reduced from 20px */
        box-shadow: var(--shadow);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px -3px rgba(0, 0, 0, 0.4);
        border-color: rgba(255,255,255,0.15);
    }
    
    .metric-label {
        font-size: 10px; /* Reduced from 13px */
        color: var(--text-secondary);
        font-weight: 500;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 21px; /* Reduced from 28px */
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }
    
    .metric-sub {
        font-size: 9px; /* Reduced from 12px */
        margin-top: 3px;
        font-weight: 500;
        opacity: 0.9;
    }
    
    /* --- STAT TILES (Smaller) --- */
    .stat-tile {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 6px; /* Reduced from 8px */
        padding: 9px; /* Reduced from 12px */
        margin-bottom: 6px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.03);
    }
    .stat-label {
        font-size: 9px; /* Reduced from 11px */
        color: var(--text-secondary);
        margin-bottom: 3px;
        font-weight: 500;
    }
    .stat-val {
        font-size: 11px; /* Reduced from 15px */
        font-weight: 700;
        color: var(--text-primary);
    }

    /* --- DATA SOURCE BOX --- */
    .data-source-container {
        background-color: var(--card-bg);
        border: var(--card-border);
        border-radius: var(--radius);
        padding: 15px 20px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* --- SCROLLBAR --- */
    ::-webkit-scrollbar {
        height: 8px; width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0e1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* --- HEADINGS --- */
    h1, h2, h3 {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    h3 { /* Specific ID title */
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    /* --- TABS --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        color: var(--text-secondary);
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: var(--accent-brand);
        border-bottom: 2px solid var(--accent-brand) !important;
    }

    /* --- HIDE HEADER ANCHORS --- */
    /* Remove link icons from headers as requested */
    .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a, .stMarkdown h4 a, .stMarkdown h5 a, .stMarkdown h6 a {
        display: none !important;
        pointer-events: none;
        color: inherit !important;
        text-decoration: none !important;
    }
    /* Older Streamlit versions might use .anchor-link */
    a.anchor-link { display: none !important; }

    </style>
""", unsafe_allow_html=True)

# --- DIALOGS ---


# --- SESSION STATE INITIALIZATION ---


if 'df_trades' not in st.session_state:
    st.session_state['df_trades'] = pd.DataFrame()
if 'df_deposits' not in st.session_state:
    st.session_state['df_deposits'] = pd.DataFrame()

if 'last_processed_file_id' not in st.session_state:
    st.session_state['last_processed_file_id'] = None



if 'filter_symbols' not in st.session_state: st.session_state['filter_symbols'] = []
if 'filter_days' not in st.session_state: st.session_state['filter_days'] = []
if 'filter_type' not in st.session_state: st.session_state['filter_type'] = []
if 'filter_outcome' not in st.session_state: st.session_state['filter_outcome'] = []

def group_deals_to_trades(df):
    """
    Groups separate deals (entry/exit) into single trades based on position_id.
    Works for both MT5 history deals and uploaded deal reports.
    """
    if df.empty: return pd.DataFrame()
    
    # Check if we have necessary columns for grouping (Case-Insensitive Check)
    # We need position_id to group deals
    col_map = {str(c).lower().replace(' ', '_'): c for c in df.columns}
    
    if 'position_id' not in col_map:
        # Maybe it's already a Trades table? (Ticket, Profit, etc)
        # If so, just return it (maybe normalize it later)
        return df
        
    # Grouping Logic - Only rename if we are actually processing deals
    df = df.copy()
    df.columns = [str(c).lower().replace(' ', '_') for c in df.columns]

    agg_funcs = {
        'time': ['min', 'max'], 
        'symbol': 'first', 
        'type': 'last', 
        'profit': 'sum', 
        'commission': 'sum', 
        'swap': 'sum', 
        'fee': 'sum', 
        'volume': 'first'
    }
    
    # Ensure columns exist
    for col in ['commission', 'swap', 'fee', 'profit', 'volume']:
        if col not in df.columns: df[col] = 0.0
            
    try:
        df_grouped = df.groupby('position_id').agg(agg_funcs).reset_index()
        df_grouped.columns = ['Ticket', 'open_time', 'close_time', 'Symbol', 'Type', 'GrossProfit', 'Commission', 'Swap', 'Fee', 'Volume']
        
        # Post-Processing
        df_grouped['open_time'] = pd.to_datetime(df_grouped['open_time'])
        df_grouped['close_time'] = pd.to_datetime(df_grouped['close_time'])
        df_grouped['Duration'] = df_grouped['close_time'] - df_grouped['open_time']
        df_grouped['time'] = df_grouped['close_time'] # Use close time for plotting
        df_grouped['Profit'] = (df_grouped['GrossProfit'] + df_grouped['Commission'] + df_grouped['Swap'] + df_grouped['Fee'])
        
        # Determine Direction
        def get_direction(row):
            t = row['Type']
            if str(t).isdigit():
                t = int(t)
                return 'Long' if t == 1 else ('Short' if t == 0 else 'Other') # Approximate
            return 'Other'

        df_grouped['Direction'] = df_grouped.apply(get_direction, axis=1)
        
        return df_grouped
    except Exception as e:
        st.error(f"Error grouping deals: {e}")
        return pd.DataFrame()

def extract_positions_table(df):
    start_row = -1
    for i, row in df.iterrows():
        row_str = row.astype(str).str.lower().tolist()
        if 'time' in row_str and 'symbol' in row_str and 'profit' in row_str:
            start_row = i
            break
    if start_row == -1:
        st.error("Could not find 'Positions' table headers.")
        return pd.DataFrame()
    df.columns = df.iloc[start_row]
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    potential_data = df[start_row + 1:].reset_index(drop=True)
    valid_rows = []
    for i, row in potential_data.iterrows():
        first_cell = str(row.iloc[0]).strip().lower()
        if 'total' in first_cell or 'orders' in first_cell or 'deals' in first_cell: break
        if row.isnull().all(): break
        valid_rows.append(row)
    return pd.DataFrame(valid_rows) if valid_rows else pd.DataFrame()

def normalize_columns(df):
    rename_map = { 'Symbol': 'Symbol', 'Profit': 'GrossProfit', 'Commission': 'Commission', 'Swap': 'Swap', 'Fee': 'Fee', 'Time': 'open_time', 'Time.1': 'close_time', 'Type': 'TypeRaw', 'Ticket': 'Ticket', 'Volume': 'Volume', 'Size': 'Volume' }
    actual_map = {}
    for col in df.columns:
        col_str = str(col).strip()
        if col_str in rename_map: actual_map[col] = rename_map[col_str]
    df.rename(columns=actual_map, inplace=True)
    if 'Ticket' not in df.columns: df['Ticket'] = df.index
    def clean_currency(series): return pd.to_numeric(series.astype(str).str.replace(' ', '').str.replace(',', ''), errors='coerce').fillna(0)
    for col in ['GrossProfit', 'Commission', 'Swap', 'Fee', 'Volume']:
        if col in df.columns: df[col] = clean_currency(df[col])
        else: df[col] = 0.0
    df['Profit'] = df['GrossProfit'] + df['Commission'] + df['Swap'] + df['Fee']
    if 'open_time' in df.columns: df['open_time'] = pd.to_datetime(df['open_time'], errors='coerce')
    if 'close_time' in df.columns: df['close_time'] = pd.to_datetime(df['close_time'], errors='coerce')
    if 'open_time' in df.columns and 'close_time' in df.columns:
        df['Duration'] = df['close_time'] - df['open_time']; df['time'] = df['close_time']
    elif 'open_time' in df.columns: df['time'] = df['open_time']; df['Duration'] = pd.Timedelta(seconds=0)
    if 'Symbol' in df.columns: df = df[df['Symbol'].notna()]; df = df[df['Symbol'].astype(str).str.strip() != '']
    if 'TypeRaw' in df.columns: df['Direction'] = df['TypeRaw'].astype(str).str.lower().map({'buy': 'Long', 'sell': 'Short'}).fillna('Other')
    else: df['Direction'] = 'Other'
    return df.dropna(subset=['Profit'])

def format_duration(td):
    if pd.isnull(td): return "00:00:00"
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def get_consecutive_stats(df):
    profits = df['Profit'].values
    if len(profits) == 0: return 0,0,0,0
    max_wins, max_loss = 0, 0
    max_win_amt, max_loss_amt = 0.0, 0.0
    curr_wins, curr_loss = 0, 0
    curr_win_amt, curr_loss_amt = 0.0, 0.0
    for p in profits:
        if p > 0:
            curr_wins += 1; curr_win_amt += p; curr_loss = 0; curr_loss_amt = 0.0
            if curr_wins > max_wins: max_wins = curr_wins; max_win_amt = curr_win_amt
            elif curr_wins == max_wins: max_win_amt = max(max_win_amt, curr_win_amt)
        elif p < 0:
            curr_loss += 1; curr_loss_amt += p; curr_wins = 0; curr_win_amt = 0.0
            if curr_loss > max_loss: max_loss = curr_loss; max_loss_amt = curr_loss_amt
            elif curr_loss == max_loss: max_loss_amt = min(max_loss_amt, curr_loss_amt)
    return max_wins, max_win_amt, max_loss, max_loss_amt

def calculate_stats(df):
    if df.empty: return None
    net_profit = df['Profit'].sum()
    total_trades = len(df)
    winning_trades = df[df['Profit'] > 0]
    losing_trades = df[df['Profit'] < 0]
    prof_trades_count = len(winning_trades)
    loss_trades_count = len(losing_trades)
    win_rate = (prof_trades_count / total_trades * 100) if total_trades > 0 else 0
    loss_rate = (loss_trades_count / total_trades * 100) if total_trades > 0 else 0
    gross_profit_val = winning_trades['Profit'].sum()
    gross_loss_val = losing_trades['Profit'].sum() 
    profit_factor = (gross_profit_val / abs(gross_loss_val)) if abs(gross_loss_val) > 0 else 0
    equity = df['Profit'].cumsum()
    max_dd = (equity.cummax() - equity).max()
    largest_profit = df['Profit'].max()
    largest_loss = df['Profit'].min()
    avg_profit = winning_trades['Profit'].mean() if not winning_trades.empty else 0
    avg_loss = losing_trades['Profit'].mean() if not losing_trades.empty else 0
    expected_payoff = net_profit / total_trades if total_trades > 0 else 0
    std_dev = df['Profit'].std()
    sharpe = (df['Profit'].mean() / std_dev) if std_dev != 0 else 0
    cons_wins, cons_win_amt, cons_loss, cons_loss_amt = get_consecutive_stats(df)
    avg_win_dur = winning_trades['Duration'].mean() if not winning_trades.empty else pd.Timedelta(0)
    avg_loss_dur = losing_trades['Duration'].mean() if not losing_trades.empty else pd.Timedelta(0)
    longs = df[df['Direction'] == 'Long']; shorts = df[df['Direction'] == 'Short']
    long_total = len(longs); long_pnl = longs['Profit'].sum(); long_wins = longs[longs['Profit'] > 0]; long_losses = longs[longs['Profit'] < 0]
    short_total = len(shorts); short_pnl = shorts['Profit'].sum(); short_wins = shorts[shorts['Profit'] > 0]; short_losses = shorts[shorts['Profit'] < 0]
    
    return {
        "Net Profit": net_profit, "Total Trades": total_trades, "Profit Factor": profit_factor,
        "Win Rate": win_rate, "Max Drawdown": max_dd,
        "Gross Profit": gross_profit_val, "Gross Loss": gross_loss_val,
        "Profitable Trades": prof_trades_count, "Loss Trades": loss_trades_count,
        "Win Rate %": win_rate, "Loss Rate %": loss_rate,
        "Largest Profit": largest_profit, "Largest Loss": largest_loss,
        "Avg Profit": avg_profit, "Avg Loss": avg_loss,
        "Avg Win Dur": format_duration(avg_win_dur), "Avg Loss Dur": format_duration(avg_loss_dur),
        "Cons Wins": cons_wins, "Cons Win Amt": cons_win_amt,
        "Cons Loss": cons_loss, "Cons Loss Amt": cons_loss_amt,
        "Expected Payoff": expected_payoff, "Sharpe": sharpe,
        "Long Total": long_total, "Long PnL": long_pnl,
        "Long WinRate": (len(long_wins)/long_total*100) if long_total else 0,
        "Long LossRate": (len(long_losses)/long_total*100) if long_total else 0,
        "Long Wins": len(long_wins), "Long Losses": len(long_losses),
        "Long Best": longs['Profit'].max() if not longs.empty else 0,
        "Long Worst": longs['Profit'].min() if not longs.empty else 0,
        "Long Avg Win": long_wins['Profit'].mean() if not long_wins.empty else 0,
        "Long Avg Loss": long_losses['Profit'].mean() if not long_losses.empty else 0,
        "Long Win Dur": format_duration(long_wins['Duration'].mean() if not long_wins.empty else pd.Timedelta(0)),
        "Long Loss Dur": format_duration(long_losses['Duration'].mean() if not long_losses.empty else pd.Timedelta(0)),
        "Short Total": short_total, "Short PnL": short_pnl,
        "Short WinRate": (len(short_wins)/short_total*100) if short_total else 0,
        "Short LossRate": (len(short_losses)/short_total*100) if short_total else 0,
        "Short Wins": len(short_wins), "Short Losses": len(short_losses),
        "Short Best": shorts['Profit'].max() if not shorts.empty else 0,
        "Short Worst": shorts['Profit'].min() if not shorts.empty else 0,
        "Short Avg Win": short_wins['Profit'].mean() if not short_wins.empty else 0,
        "Short Avg Loss": short_losses['Profit'].mean() if not short_losses.empty else 0,
        "Short Win Dur": format_duration(short_wins['Duration'].mean() if not short_wins.empty else pd.Timedelta(0)),
        "Short Loss Dur": format_duration(short_losses['Duration'].mean() if not short_losses.empty else pd.Timedelta(0)),
        "Short Avg Win": short_wins['Profit'].mean() if not short_wins.empty else 0,
        "Short Avg Loss": short_losses['Profit'].mean() if not short_losses.empty else 0,
        "Short Win Dur": format_duration(short_wins['Duration'].mean() if not short_wins.empty else pd.Timedelta(0)),
        "Short Loss Dur": format_duration(short_losses['Duration'].mean() if not short_losses.empty else pd.Timedelta(0)),
        "Short Loss Dur": format_duration(short_losses['Duration'].mean() if not short_losses.empty else pd.Timedelta(0)),
        "Total Commission": df['Commission'].sum() if 'Commission' in df.columns else 0.0,
        "Total Swap": df['Swap'].sum() if 'Swap' in df.columns else 0.0,
    }

def get_color_style(val_num):
    if val_num > 0: return "color: var(--accent-success);"
    if val_num < 0: return "color: var(--accent-danger);"
    return "color: var(--text-secondary);"

def kpi_card_html(label, value, raw_val, sub_label=None):
    style = get_color_style(raw_val)
    sub_html = f'<div class="metric-sub" style="{style}">{sub_label}</div>' if sub_label else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="{style}">{value}</div>
        {sub_html}
    </div>
    """

def stat_card_html(label, value, raw_val=None, is_bold=False):
    style = ""
    if raw_val is not None: style = get_color_style(raw_val)
    return f"""
    <div class="stat-tile">
        <span class="stat-label">{label}</span>
        <span class="stat-val" style="{style}">{value}</span>
    </div>
    """

def format_currency(val):
    if val < 0: return f"-${abs(val):,.2f}"
    return f"${val:,.2f}"

def render_calendar_view(df):
    if df.empty: return
    df['date'] = df['time'].dt.date
    count_col = 'Ticket' if 'Ticket' in df.columns else 'Profit'
    daily_stats = df.groupby('date').agg({'Profit': 'sum', count_col: 'count'}).reset_index()
    if count_col != 'Ticket': daily_stats.rename(columns={count_col: 'Ticket'}, inplace=True)
    daily_stats.set_index('date', inplace=True)
    df = df.sort_values(by='time', ascending=False)
    df['YearMonth'] = df['time'].dt.to_period('M')
    unique_months = df['YearMonth'].unique()
    today_date = datetime.now().date()
    st.subheader("Trading Calendar")
    html_content = "<div style='display: flex; flex-direction: row-reverse; overflow-x: auto; padding-bottom: 20px;'>"
    for ym in unique_months:
        year = ym.year; month = ym.month; month_name = calendar.month_name[month]
        month_data = df[df['YearMonth'] == ym]
        m_pnl = month_data['Profit'].sum(); m_trades = len(month_data)
        m_color = "var(--accent-success)" if m_pnl >= 0 else "var(--accent-danger)"
        m_pnl_str = f"${m_pnl:,.2f}" if m_pnl >= 0 else f"-${abs(m_pnl):,.2f}"
        
        cal = calendar.Calendar(firstweekday=0); month_days = cal.monthdayscalendar(year, month)
        num_weeks = len(month_days); grid_template = f"30px repeat({num_weeks}, 85px)"
        
        grid_items_html = ""
        # Header row (Day Names) - Empty corner
        grid_items_html += '<div></div>' 
        # Removed top "Week" headers as requested
        for i in range(num_weeks): grid_items_html += f'<div></div>'
        
        day_labels = ["M", "T", "W", "T", "F", "S", "S"]
        weekly_pnl = [0.0] * num_weeks; weekly_trades = [0] * num_weeks
        
        for day_idx, day_name in enumerate(day_labels):
            grid_items_html += f'<div style="font-size:11px; font-weight:bold; color:var(--text-secondary); display:flex; align-items:center; justify-content:center;">{day_name}</div>'
            for week_idx, week in enumerate(month_days):
                day_num = week[day_idx]
                if day_num == 0:
                    grid_items_html += '<div style="visibility: hidden; height: 60px; margin: 2px;"></div>'; continue
                    
                current_date = datetime(year, month, day_num).date()
                day_pnl = 0.0; day_count = 0; bg_color = "transparent"; border_style = "1px solid var(--card-border)"; val_str = ""; display_val_color = "var(--text-secondary)"
                
                # Active Day Style
                if current_date in daily_stats.index:
                    day_pnl = daily_stats.loc[current_date, 'Profit']
                    day_count = int(daily_stats.loc[current_date, 'Ticket'])
                    weekly_pnl[week_idx] += day_pnl; weekly_trades[week_idx] += day_count
                    
                    if day_pnl > 0: 
                        bg_color = "rgba(0, 210, 106, 0.15)"; 
                        border_style = "1px solid rgba(0, 210, 106, 0.3)";
                        display_val_color = "var(--accent-success)"
                    elif day_pnl < 0: 
                        bg_color = "rgba(248, 49, 47, 0.15)"; 
                        border_style = "1px solid rgba(248, 49, 47, 0.3)";
                        display_val_color = "var(--accent-danger)"
                    
                    if day_count > 0: 
                        val_str = f"${day_pnl:,.2f}" if day_pnl >= 0 else f"-${abs(day_pnl):,.2f}"
                elif current_date < today_date:
                    # Past empty day
                    bg_color = "rgba(255, 255, 255, 0.02)"
                
                # Today Highlight
                if current_date == today_date: border_style = "2px solid var(--accent-brand)"

                count_html = f"<div style='font-size: 9px; color: var(--text-secondary); text-align: right;'>{day_count} Trades</div>" if day_count > 0 else "<div style='height: 11px;'></div>"

                grid_items_html += f'<div style="background-color: {bg_color}; border: {border_style}; border-radius: 6px; height: 60px; margin: 2px; padding: 4px; display: flex; flex-direction: column; justify-content: space-between; transition: all 0.2s;"><div style="font-size: 9px; font-weight: bold; color: var(--text-secondary);">{day_num}</div><div style="text-align: right; font-weight: 700; font-size: 11px; color: {display_val_color}; line-height: 1.1;">{val_str}</div>{count_html}</div>'
                
        # Weekly Totals Row
        grid_items_html += '<div style="font-size:9px; font-weight:bold; color:var(--text-secondary); display:flex; align-items:center; justify-content:center;">Total</div>'
        for i in range(num_weeks):
            w_color = "var(--accent-success)" if weekly_pnl[i] >= 0 else ("var(--accent-danger)" if weekly_pnl[i] < 0 else "var(--text-secondary)"); 
            w_val = weekly_pnl[i]
            w_str = (f"${w_val:,.2f}" if w_val >= 0 else f"-${abs(w_val):,.2f}") if weekly_trades[i] > 0 else "-"
            w_count_str = f"{weekly_trades[i]} Trades" if weekly_trades[i] > 0 else ""
            
            # Weekly Total Card Style
            w_bg = "rgba(0,0,0,0.3)" # Darker background
            
            grid_items_html += f'<div style="background-color: {w_bg}; border: 1px solid var(--card-border); border-radius: 6px; height: 60px; margin: 2px; padding: 4px; display: flex; flex-direction: column; justify-content: space-between; align-items: center;"><div style="font-size:9px; font-weight:bold; color:#777; width:100%; text-align:center;">Week {i+1}</div><div style="font-weight: 800; font-size: 11px; color: {w_color};">{w_str}</div><div style="font-size: 8px; color: var(--text-secondary); margin-top:2px;">{w_count_str}</div></div>'
            
        month_card = f"<div style='width: fit-content; background-color: var(--card-bg); border: 1px solid var(--card-border); border-radius: var(--radius); padding: 15px; margin-right: 15px;'><div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'><h4 style='margin: 0; font-size: 14px; color: var(--text-primary); font-weight: 600; text-transform: uppercase;'>{month_name} {year}</h4><div style='text-align: right;'><span style='font-size: 14px; font-weight: bold; color: {m_color};'>{m_pnl_str}</span><span style='font-size: 11px; color: var(--text-secondary); margin-left: 5px;'>({m_trades} Trades)</span></div></div><div style='display: grid; grid-template-columns: {grid_template};'>{grid_items_html}</div></div>"
        html_content += month_card
    html_content += "</div>"
    st.markdown(html_content, unsafe_allow_html=True)

def reset_filters():
    st.session_state['filter_symbols'] = []; st.session_state['filter_days'] = []
    st.session_state['filter_type'] = []; st.session_state['filter_outcome'] = []
    st.session_state['filter_reverse'] = False
    st.session_state['filter_reverse'] = False
    if 'filter_dates' in st.session_state: del st.session_state['filter_dates']
    if 'filter_volume_min' in st.session_state: del st.session_state['filter_volume_min']
    if 'filter_volume_max' in st.session_state: del st.session_state['filter_volume_max']

# --- MAIN UI LAYOUT ---

st.markdown("<h3>📊 Trade Analysis Dashboard</h3>", unsafe_allow_html=True)

# 1. FILE UPLOADER (TOP OF MAIN)
with st.expander("📂 Upload Data", expanded=st.session_state['df_trades'].empty):
    uploaded_file = st.file_uploader("Upload Report (.xlsx)", type=['xlsx'], label_visibility="collapsed")
    
    if uploaded_file:
         file_id = f"{uploaded_file.name}-{uploaded_file.size}"
         if st.session_state.get('last_processed_file_id') != file_id:
             with st.spinner("Processing Trade Data..."):
                 try:
                     # 1. Read Raw
                     raw_df = pd.read_excel(uploaded_file, header=None)
                     
                     # 2. Extract Table (using existing helper)
                     df_temp = extract_positions_table(raw_df)
                     
                     # Fallback if extraction failed (maybe standard table)
                     if df_temp.empty:
                         # Try re-reading with headers
                         uploaded_file.seek(0)
                         df_temp = pd.read_excel(uploaded_file)
                     
                     # 3. Clean Columns (using existing helper)
                     df_clean = normalize_columns(df_temp)
                     
                     # 4. Group Deals (if applicable)
                     # Pass to grouping logic which handles both Checks
                     df_final = group_deals_to_trades(df_clean)
                     
                     # 5. Store
                     st.session_state['df_trades'] = df_final
                     st.session_state['last_processed_file_id'] = file_id
                     st.success(f"Success! {len(df_final)} trades loaded.")
                     st.rerun()
                     
                 except Exception as e:
                     st.error(f"Error parsing file: {e}")

if st.session_state['df_trades'].empty:
    st.info("👋 Welcome! Please upload your Trade Report (.xlsx) above to begin analysis.")
    st.stop()


# --- FILTER SIDEBAR ---
df_trades = st.session_state['df_trades']
df_filtered = df_trades.copy()

if not df_trades.empty:
    with st.sidebar:
        st.markdown("### 🔎 Filters")
        reverse_mode = st.checkbox("Reverse Trades", value=st.session_state.get('filter_reverse', False), key="filter_reverse")
        st.divider()
        min_date = df_trades['time'].min().date(); max_date = df_trades['time'].max().date()
        date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date, format="DD/MM/YYYY", key="filter_dates")
        available_symbols = sorted(df_trades['Symbol'].unique())
        selected_symbols = st.multiselect("Symbols", available_symbols, key="filter_symbols")
        
        # Lot Size Filter
        if 'Volume' in df_trades.columns:
            min_vol = float(df_trades['Volume'].min())
            max_vol = float(df_trades['Volume'].max())
            if min_vol == max_vol: max_vol += 0.1
            if min_vol == max_vol: max_vol += 0.1
            st.markdown("<p style='font-size: 13px; font-weight: 600; margin-bottom: 2px;'>Lot Size Range</p>", unsafe_allow_html=True)
            v1, v2 = st.columns(2)
            with v1: v_min_input = st.number_input("Min", min_value=0.0, value=min_vol, step=0.01, key="filter_volume_min")
            with v2: v_max_input = st.number_input("Max", min_value=0.0, value=max_vol, step=0.01, key="filter_volume_max")
        
        # Spacer for visual balance
        if 'Volume' in df_trades.columns: st.write("")
        
        df_trades['DayName'] = df_trades['time'].dt.day_name()
        available_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        existing_days = [d for d in available_days if d in df_trades['DayName'].unique()]
        selected_days = st.multiselect("Weekdays", existing_days, key="filter_days")
        selected_type = st.multiselect("Type", ['Long', 'Short'], key="filter_type")
        selected_outcome = st.multiselect("Outcome", ['Win', 'Loss'], key="filter_outcome")
        st.write(""); st.write("")
        st.button("Reset Filters", on_click=reset_filters)

    if reverse_mode:
        df_filtered['GrossProfit'] = -df_filtered['GrossProfit']
        df_filtered['Profit'] = df_filtered['GrossProfit'] + df_filtered['Commission'] + df_filtered['Swap'] + df_filtered['Fee']
        df_filtered['Direction'] = df_filtered['Direction'].map({'Long': 'Short', 'Short': 'Long', 'Other': 'Other'})

    if date_range:
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_d, end_d = date_range
            df_filtered = df_filtered[(df_filtered['time'].dt.date >= start_d) & (df_filtered['time'].dt.date <= end_d)]
        elif isinstance(date_range, tuple) and len(date_range) == 1:
             start_d = date_range[0]
             df_filtered = df_filtered[df_filtered['time'].dt.date >= start_d]
    if selected_symbols: df_filtered = df_filtered[df_filtered['Symbol'].isin(selected_symbols)]
    
    # Apply Volume Filter
    if 'Volume' in df_filtered.columns:
        v_min = st.session_state.get('filter_volume_min', 0.0)
        v_max = st.session_state.get('filter_volume_max', 100000.0) # sufficiently large default if not set
        df_filtered = df_filtered[(df_filtered['Volume'] >= v_min) & (df_filtered['Volume'] <= v_max)]
        
    if 'DayName' not in df_filtered.columns: df_filtered['DayName'] = df_filtered['time'].dt.day_name()
    if selected_days: df_filtered = df_filtered[df_filtered['DayName'].isin(selected_days)]
    if selected_type: df_filtered = df_filtered[df_filtered['Direction'].isin(selected_type)]
    if selected_outcome:
        if 'Win' in selected_outcome and 'Loss' in selected_outcome: pass
        elif 'Win' in selected_outcome: df_filtered = df_filtered[df_filtered['Profit'] > 0]
        elif 'Loss' in selected_outcome: df_filtered = df_filtered[df_filtered['Profit'] < 0]

# --- MAIN TABS ---
if not df_filtered.empty:
    if 'Direction' not in df_filtered.columns: df_filtered['Direction'] = 'Other'
    df_filtered = df_filtered.sort_values(by='time')
    df_filtered['Equity_Curve'] = df_filtered['Profit'].cumsum()
    stats = calculate_stats(df_filtered)
    
    # Dashboard View
    with st.container():
        if stats:
            st.subheader("Stats")
            kpi_cols = st.columns(5)
            kpis = [
                ("Net Profit", format_currency(stats['Net Profit']), stats['Net Profit']),
                ("Total Trades", str(stats['Total Trades']), 0),
                ("Profit Factor", f"{stats['Profit Factor']:.2f}", stats['Profit Factor'] - 1),
                ("Win Rate", f"{stats['Win Rate']:.1f}%", stats['Win Rate'] - 50),
                ("Max Drawdown", format_currency(stats['Max Drawdown']), -1)
            ]
            for i, col in enumerate(kpi_cols):
                with col:
                    st.markdown(kpi_card_html(kpis[i][0], kpis[i][1], kpis[i][2]), unsafe_allow_html=True)
            st.write("") 

            d1, d2, d3 = st.columns(3)
            with d1:
                with st.container(border=True):
                    st.markdown("<h6 style='text-align: center; margin: 0 0 10px 0;'>General Stats</h6>", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1: st.markdown(stat_card_html("Gross Profit", format_currency(stats['Gross Profit']), stats['Gross Profit'], True), unsafe_allow_html=True)
                    with c2: st.markdown(stat_card_html("Gross Loss", format_currency(stats['Gross Loss']), stats['Gross Loss'], True), unsafe_allow_html=True)
                    c3, c4 = st.columns(2)
                    with c3: st.markdown(stat_card_html("Profit Trades", f"{stats['Profitable Trades']} ({stats['Win Rate %']:.1f}%)", 1), unsafe_allow_html=True)
                    with c4: st.markdown(stat_card_html("Loss Trades", f"{stats['Loss Trades']} ({stats['Loss Rate %']:.1f}%)", -1), unsafe_allow_html=True)
                    c5, c6 = st.columns(2)
                    with c5: 
                        st.markdown(stat_card_html("Largest Profit", format_currency(stats['Largest Profit']), 1), unsafe_allow_html=True)
                        st.markdown(stat_card_html("Average Profit", format_currency(stats['Avg Profit']), 1), unsafe_allow_html=True)
                    with c6:
                        st.markdown(stat_card_html("Largest Loss", format_currency(stats['Largest Loss']), -1), unsafe_allow_html=True)
                        st.markdown(stat_card_html("Average Loss", format_currency(stats['Avg Loss']), -1), unsafe_allow_html=True)
                    c7, c8 = st.columns(2)
                    with c7: st.markdown(stat_card_html("Avg Win Duration", stats['Avg Win Dur']), unsafe_allow_html=True)
                    with c8: st.markdown(stat_card_html("Avg Loss Duration", stats['Avg Loss Dur']), unsafe_allow_html=True)
                    c9, c10 = st.columns(2)
                    with c9: st.markdown(stat_card_html("Consecutive Wins", f"{stats['Cons Wins']} ({format_currency(stats['Cons Win Amt'])})", 1), unsafe_allow_html=True)
                    with c10: st.markdown(stat_card_html("Consecutive Losses", f"{stats['Cons Loss']} ({format_currency(stats['Cons Loss Amt'])})", -1), unsafe_allow_html=True)
                    c11, c12 = st.columns(2)
                    with c11: st.markdown(stat_card_html("Expected Payoff", f"{stats['Expected Payoff']:.2f}", stats['Expected Payoff']), unsafe_allow_html=True)
                    with c12: st.markdown(stat_card_html("Sharpe Ratio", f"{stats['Sharpe']:.2f}", stats['Sharpe']), unsafe_allow_html=True)
                    c13, c14 = st.columns(2)
                    with c13: st.markdown(stat_card_html("Total Commission", format_currency(stats['Total Commission']), stats['Total Commission']), unsafe_allow_html=True)
                    with c14: st.markdown(stat_card_html("Total Swap", format_currency(stats['Total Swap']), stats['Total Swap']), unsafe_allow_html=True)

            with d2:
                with st.container(border=True):
                    st.markdown("<h6 style='text-align: center; margin: 0 0 10px 0;'>Long Stats (Buy)</h6>", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1: st.markdown(stat_card_html("Total Buys", str(stats['Long Total'])), unsafe_allow_html=True)
                    with c2: st.markdown(stat_card_html("Buy PnL", format_currency(stats['Long PnL']), stats['Long PnL'], True), unsafe_allow_html=True)
                    c3, c4 = st.columns(2)
                    with c3: st.markdown(stat_card_html("Win Rate", f"{stats['Long WinRate']:.1f}% ({stats['Long Wins']})", 1), unsafe_allow_html=True)
                    with c4: st.markdown(stat_card_html("Loss Rate", f"{stats['Long LossRate']:.1f}% ({stats['Long Losses']})", -1), unsafe_allow_html=True)
                    c5, c6 = st.columns(2)
                    with c5: st.markdown(stat_card_html("Best Trade", format_currency(stats['Long Best']), 1), unsafe_allow_html=True)
                    with c6: st.markdown(stat_card_html("Worst Trade", format_currency(stats['Long Worst']), -1), unsafe_allow_html=True)
                    c7, c8 = st.columns(2)
                    with c7: st.markdown(stat_card_html("Avg Win", format_currency(stats['Long Avg Win']), 1), unsafe_allow_html=True)
                    with c8: st.markdown(stat_card_html("Avg Loss", format_currency(stats['Long Avg Loss']), -1), unsafe_allow_html=True)
                    c9, c10 = st.columns(2)
                    with c9: st.markdown(stat_card_html("Avg Win Duration", stats['Long Win Dur']), unsafe_allow_html=True)
                    with c10: st.markdown(stat_card_html("Avg Loss Duration", stats['Long Loss Dur']), unsafe_allow_html=True)

            with d3:
                with st.container(border=True):
                    st.markdown("<h6 style='text-align: center; margin: 0 0 10px 0;'>Short Stats (Sell)</h6>", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1: st.markdown(stat_card_html("Total Sells", str(stats['Short Total'])), unsafe_allow_html=True)
                    with c2: st.markdown(stat_card_html("Sell PnL", format_currency(stats['Short PnL']), stats['Short PnL'], True), unsafe_allow_html=True)
                    c3, c4 = st.columns(2)
                    with c3: st.markdown(stat_card_html("Win Rate", f"{stats['Short WinRate']:.1f}% ({stats['Short Wins']})", 1), unsafe_allow_html=True)
                    with c4: st.markdown(stat_card_html("Loss Rate", f"{stats['Short LossRate']:.1f}% ({stats['Short Losses']})", -1), unsafe_allow_html=True)
                    c5, c6 = st.columns(2)
                    with c5: st.markdown(stat_card_html("Best Trade", format_currency(stats['Short Best']), 1), unsafe_allow_html=True)
                    with c6: st.markdown(stat_card_html("Worst Trade", format_currency(stats['Short Worst']), -1), unsafe_allow_html=True)
                    c7, c8 = st.columns(2)
                    with c7: st.markdown(stat_card_html("Avg Win", format_currency(stats['Short Avg Win']), 1), unsafe_allow_html=True)
                    with c8: st.markdown(stat_card_html("Avg Loss", format_currency(stats['Short Avg Loss']), -1), unsafe_allow_html=True)
                    c9, c10 = st.columns(2)
                    with c9: st.markdown(stat_card_html("Avg Win Duration", stats['Short Win Dur']), unsafe_allow_html=True)
                    with c10: st.markdown(stat_card_html("Avg Loss Duration", stats['Short Loss Dur']), unsafe_allow_html=True)

            st.divider()
            render_calendar_view(df_filtered)
            st.divider()

            st.subheader("Graphs")
            
            # Common Layout for consistent look
            common_layout = dict(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E0E0E0', family='Inter'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)'),
                margin=dict(l=20, r=20, t=50, b=20)
            )

            # Prepare Balance Data (Merge Trades and Deposits)
            fig_cum = go.Figure()
            
            df_balance_curve = pd.DataFrame()
            if not df_filtered.empty:
                temp_trades = df_filtered[['time', 'Profit']].copy()
                temp_trades['Type'] = 'Trade'
            else: temp_trades = pd.DataFrame(columns=['time', 'Profit', 'Type'])
            
            df_deps = st.session_state.get('df_deposits', pd.DataFrame())
            
            if not df_deps.empty:
                # Filter deposits by date range if applicable? usually balance curve should account for all history to be accurate
                # but if user filters date, they might expect to see balance impact during that period.
                # However, "Balance" implies accumulated total. If we filter start date, do we start from 0 or actual balance?
                # User request: "use balance instead of profit". 
                # Let's include ALL deposits to establish baseline, but only calculate curve segments?
                # Simplest approach: Merge all, calculate running balance, then crop to view.
                
                temp_deps = df_deps[['time', 'Amount']].rename(columns={'Amount': 'Profit'})
                temp_deps['Type'] = 'Deposit'
                
                combined_ops = pd.concat([temp_trades, temp_deps])
                combined_ops.sort_values('time', inplace=True)
                combined_ops['Balance'] = combined_ops['Profit'].cumsum()
                
                # Plot
                fig_cum.add_trace(go.Scatter(
                    x=combined_ops['time'], 
                    y=combined_ops['Balance'], 
                    mode='lines', 
                    name='Balance', 
                    line=dict(color='#00D26A', width=4), # Enhanced width for visibility
                    fill='tozeroy',
                    fillcolor="rgba(0, 210, 106, 0.2)" # Slightly more opacity
                ))
                
                # Show Deposits markers
                deposits_only = combined_ops[combined_ops['Type'] == 'Deposit']
                if not deposits_only.empty:
                    fig_cum.add_trace(go.Scatter(
                        x=deposits_only['time'],
                        y=deposits_only['Balance'],
                        mode='markers',
                        name='Deposits',
                        marker=dict(symbol='triangle-up', size=10, color='yellow')
                    ))
                
                pass

            else:
                # Fallback to cumulative profit if no deposits
                line_color = '#00D26A' if df_filtered['Equity_Curve'].iloc[-1] >= 0 else '#F8312F'
                fig_cum.add_trace(go.Scatter(
                    x=df_filtered['time'], 
                    y=df_filtered['Equity_Curve'], 
                    mode='lines', 
                    name='Profit', 
                    line=dict(color=line_color, width=3),
                    fill='tozeroy',
                    fillcolor=f"rgba{tuple(int(line_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}"
                ))
            
            with st.container(border=True):
                fig_cum.update_layout(**common_layout, title="Account Balance" if not df_deps.empty else "Cumulative Profit", xaxis_title="Date", yaxis_title="Balance ($)", hovermode="x")
                st.plotly_chart(fig_cum, use_container_width=True)
            
            daily_df = df_filtered.copy()
            daily_df['Date'] = daily_df['time'].dt.date
            daily_df = daily_df.groupby('Date')['Profit'].sum().reset_index()
            daily_df['Color'] = daily_df['Profit'].apply(lambda x: '#00D26A' if x >= 0 else '#F8312F')
            
            with st.container(border=True):
                fig_daily = go.Figure()
                fig_daily.add_trace(go.Bar(x=daily_df['Date'], y=daily_df['Profit'], marker_color=daily_df['Color'], name='Daily PnL'))
                fig_daily.update_layout(**common_layout, title="Daily Profit/Loss", xaxis_title="Date", yaxis_title="Daily PnL ($)")
                fig_daily.update_traces(hovertemplate='Date: %{x|%d %b %y}<br>PnL: $%{y:.2f}')
                st.plotly_chart(fig_daily, use_container_width=True)
            
            g1, g2 = st.columns(2)
            with g1:
                with st.container(border=True):
                    df_filtered['Weekday'] = df_filtered['time'].dt.day_name()
                    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    weekday_stats = df_filtered.groupby('Weekday')['Profit'].sum().reindex(days_order).fillna(0).reset_index()
                    weekday_stats['Color'] = weekday_stats['Profit'].apply(lambda x: '#00D26A' if x >= 0 else '#F8312F')
                    fig_wd = go.Figure()
                    fig_wd.add_trace(go.Bar(x=weekday_stats['Weekday'], y=weekday_stats['Profit'], marker_color=weekday_stats['Color'], name='Weekday PnL'))
                    fig_wd.update_layout(**common_layout, title="Performance by Weekday", xaxis_title="Day of Week", yaxis_title="Total PnL ($)")
                    fig_wd.update_traces(hovertemplate='Day: %{x}<br>PnL: $%{y:.2f}')
                    st.plotly_chart(fig_wd, use_container_width=True)
            with g2:
                with st.container(border=True):
                    symbol_stats = df_filtered.groupby('Symbol')['Profit'].sum().reset_index().sort_values(by='Profit', ascending=False)
                    symbol_stats['Color'] = symbol_stats['Profit'].apply(lambda x: '#00D26A' if x >= 0 else '#F8312F')
                    fig_sym = go.Figure()
                    fig_sym.add_trace(go.Bar(x=symbol_stats['Symbol'], y=symbol_stats['Profit'], marker_color=symbol_stats['Color'], name='Symbol PnL'))
                    fig_sym.update_layout(**common_layout, title="Performance by Symbol", xaxis_title="Symbol", yaxis_title="Total PnL ($)")
                    fig_sym.update_traces(hovertemplate='Symbol: %{x}<br>PnL: $%{y:.2f}')
                    st.plotly_chart(fig_sym, use_container_width=True)

            # Balance vs Drawdown Graph
            with st.container(border=True):
                # Heading Removed
                
                if not df_deps.empty and not combined_ops.empty:
                    curve = combined_ops['Balance']
                    times = combined_ops['time']
                else:
                    curve = df_filtered['Equity_Curve']
                    times = df_filtered['time']
                    
                peak = curve.cummax()
                drawdown = curve - peak
                
                fig_dd = go.Figure()
                # Balance Line (Left Y)
                fig_dd.add_trace(go.Scatter(
                    x=times, y=curve, 
                    name='Balance', 
                    mode='lines',
                    line=dict(color='#00D26A', width=2)
                ))
                # Drawdown Area (Right Y)
                fig_dd.add_trace(go.Scatter(
                    x=times, y=drawdown, 
                    name='Drawdown', 
                    fill='tozeroy', 
                    mode='lines',
                    line=dict(color='#F8312F', width=1), 
                    yaxis='y2',
                    fillcolor='rgba(248, 49, 47, 0.2)' 
                ))
                
                fig_dd.update_layout(**common_layout)
                fig_dd.update_layout(
                    title="Balance vs Drawdown",
                    xaxis_title="Date",
                    yaxis=dict(title="Balance ($)", side="left", showgrid=True, zeroline=False),
                    yaxis2=dict(
                        title="Drawdown ($)", 
                        side="right", 
                        overlaying="y", 
                        showgrid=False,
                        zeroline=False
                    ),
                    legend=dict(orientation="h", y=1.1)
                )
                st.plotly_chart(fig_dd, use_container_width=True)

            # Historical Deposits (Moved to Bottom)
            if not st.session_state.get('df_deposits', pd.DataFrame()).empty:
                # Re-calculate deposits_only if needed or access from local scope if available.
                # Since we are in the same function scope, we can reconstruct or reuse.
                # df_deps was defined above.
                if not df_deps.empty:
                     # Re-filtering for logic safety
                     temp_deps_plot = df_deps.copy()
                     # make sure time is datetime
                     if 'time' in temp_deps_plot.columns:
                         temp_deps_plot['time'] = pd.to_datetime(temp_deps_plot['time'])
                     
                     with st.container(border=True):
                        # Heading Removed
                        fig_dep = go.Bar(x=temp_deps_plot['time'], y=temp_deps_plot['Amount'], name="Deposit Amount", marker_color='yellow')
                        fig_dep_layout = go.Figure(data=[fig_dep])
                        fig_dep_layout.update_layout(**common_layout, title="Deposits over Time", yaxis_title="Amount ($)")
                        st.plotly_chart(fig_dep_layout, use_container_width=True)




else:
    if not df_trades.empty:
        st.warning("No trades match the selected filters.")