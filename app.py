import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import sqlite3
import uuid
from datetime import datetime

# ── PALETTE ──────────────────────────────────────────────────────────────────
JAPANESE_PALETTE = {
    "Background": "#000B00",
    "Accent":     "#180614",
    "Primary":    "#250D00",
    "Header":     "#000B00",
    "Text":       "#F3F3F2",
}

st.set_page_config(
    page_title="Financial Volatility Surface Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
    <style>
    .stApp {{ background-color: {JAPANESE_PALETTE['Background']}; color: {JAPANESE_PALETTE['Text']}; }}
    .main-header {{ color: {JAPANESE_PALETTE['Text']}; border-bottom: 2px solid {JAPANESE_PALETTE['Header']}; padding-bottom: 10px; }}
    .metric-card {{ background-color: {JAPANESE_PALETTE['Primary']}; padding: 15px; border-radius: 10px; border-left: 4px solid {JAPANESE_PALETTE['Accent']}; }}
    </style>
""", unsafe_allow_html=True)


# ── DATABASE ──────────────────────────────────────────────────────────────────
DB_PATH = "options_log.db"

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    # Inputs table: one row per calculation
    cur.execute("""
        CREATE TABLE IF NOT EXISTS inputs (
            calc_id       TEXT PRIMARY KEY,
            timestamp     TEXT,
            spot          REAL,
            strike        REAL,
            expiry        REAL,
            rate          REAL,
            volatility    REAL,
            dividend      REAL,
            call_price    REAL,
            put_price     REAL,
            call_purchase REAL,
            put_purchase  REAL
        )
    """)
    # Outputs table: one row per (spot_shock, vol_shock) cell
    cur.execute("""
        CREATE TABLE IF NOT EXISTS outputs (
            output_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            calc_id    TEXT,
            spot_shock REAL,
            vol_shock  REAL,
            call_value REAL,
            put_value  REAL,
            call_pnl   REAL,
            put_pnl    REAL,
            FOREIGN KEY (calc_id) REFERENCES inputs(calc_id)
        )
    """)
    con.commit()
    con.close()

def save_calculation(calc_id, S, K, T, r, sigma, div,
                     call_price, put_price, call_purchase, put_purchase,
                     spot_grid, vol_grid,
                     call_surface, put_surface,
                     call_pnl_surface, put_pnl_surface):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO inputs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (calc_id, datetime.now().isoformat(), S, K, T, r, sigma, div,
          call_price, put_price, call_purchase, put_purchase))

    rows = []
    for i in range(spot_grid.shape[0]):
        for j in range(spot_grid.shape[1]):
            rows.append((
                calc_id,
                round(spot_grid[i, j] - S, 4),
                round(vol_grid[i, j] - sigma, 4),
                round(call_surface[i, j], 4),
                round(put_surface[i, j], 4),
                round(call_pnl_surface[i, j], 4),
                round(put_pnl_surface[i, j], 4),
            ))
    cur.executemany("""
        INSERT INTO outputs (calc_id, spot_shock, vol_shock, call_value, put_value, call_pnl, put_pnl)
        VALUES (?,?,?,?,?,?,?)
    """, rows)
    con.commit()
    con.close()

def load_history():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """SELECT calc_id, timestamp, spot, strike, expiry, rate,
                  volatility, call_price, put_price
           FROM inputs ORDER BY timestamp DESC LIMIT 20""",
        con,
    )
    con.close()
    return df

init_db()


# ── MODELS ────────────────────────────────────────────────────────────────────
def black_scholes(S, K, T, r, sigma, s_sigma=0.0, option_type="call"):
    d1 = (np.log(S / K) + (r - s_sigma + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def monte_carlo_simulation(S, K, T, r, s_sigma, sigma,
                           n_paths=1000, n_steps=252, option_type="call"):
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S
    z = np.random.standard_normal((n_paths, n_steps))
    h = T / n_steps
    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - s_sigma - 0.5 * sigma**2) * h + sigma * np.sqrt(h) * z[:, t - 1]
        )
    payoffs = (np.maximum(paths[:, -1] - K, 0) if option_type == "call"
               else np.maximum(K - paths[:, -1], 0))
    return np.exp(-r * T) * np.mean(payoffs), paths


# ── P&L HEATMAP HELPER ───────────────────────────────────────────────────────
def pnl_colormap():
    return mcolors.LinearSegmentedColormap.from_list(
        "pnl", ["#C62828", "#FFFFFF", "#2E7D32"]
    )

def plot_pnl_heatmap(ax, spot_prices, volatilities, pnl_surface, title, purchase_price):
    cmap    = pnl_colormap()
    abs_max = max(abs(pnl_surface.min()), abs(pnl_surface.max()), 0.01)
    norm_c  = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    ax.imshow(
        pnl_surface,
        aspect="auto",
        origin="lower",
        extent=[spot_prices[0], spot_prices[-1],
                volatilities[0] * 100, volatilities[-1] * 100],
        cmap=cmap,
        norm=norm_c,
    )

    # Annotate each cell with the P&L value
    for i in range(pnl_surface.shape[0]):
        for j in range(pnl_surface.shape[1]):
            val   = pnl_surface[i, j]
            color = "black" if abs(val) < abs_max * 0.5 else "white"
            ax.text(
                spot_prices[j],
                volatilities[i] * 100,
                f"{val:+.2f}",
                ha="center", va="center",
                fontsize=6.5, color=color, fontweight="bold",
            )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_c)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="P&L ($)")

    ax.set_xlabel("Spot Price ($)", fontsize=10)
    ax.set_ylabel("Implied Volatility (%)", fontsize=10)
    ax.set_title(f"{title}  |  Purchased at ${purchase_price:.2f}", fontsize=11)
    ax.set_facecolor("#0a0a0a")


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters & Control Panel")

    st.subheader("Model Configuration")
    model_choice = st.selectbox("Pricing Model", ["Black-Scholes", "Monte Carlo"])

    st.subheader("Base Scenario")
    S              = st.slider("Spot Price",             50.0,  150.0, 100.0, 5.0)
    K              = st.slider("Strike Price",            50.0,  150.0, 100.0, 5.0)
    T              = st.slider("Time to Expiry (Years)",  0.1,     2.0,   1.0, 0.1)
    r              = st.slider("Risk-Free Rate (%)",      0.0,    10.0,   5.0, 0.5) / 100
    sigma          = st.slider("Volatility (%)",         10.0,   100.0,  30.0, 5.0) / 100
    dividend_yield = st.slider("Dividend Yield (%)",      0.0,    10.0,   0.0, 0.5) / 100

    if model_choice == "Monte Carlo":
        st.subheader("Advanced Settings")
        n_paths = st.slider("Number of Paths", 100, 10000, 1000, 100)
        n_steps = st.slider("Number of Steps",  10,   500,  252,  10)
    else:
        st.subheader("Heatmap Shock Ranges")
        vol_min  = st.slider("Min Volatility (%)",  10.0,  50.0,  15.0, 5.0) / 100
        vol_max  = st.slider("Max Volatility (%)",  30.0, 100.0,  60.0, 5.0) / 100
        spot_min = st.slider("Min Spot Price", S * 0.5, S,       S * 0.7, 5.0)
        spot_max = st.slider("Max Spot Price", S,       S * 1.5, S * 1.3, 5.0)

    # P&L inputs
    st.subheader("P&L Analysis")
    st.caption("Enter what you paid for each option to see your P&L across scenarios.")
    call_purchase = st.number_input("Call Purchase Price ($)", min_value=0.0, value=10.0, step=0.5)
    put_purchase  = st.number_input("Put Purchase Price ($)",  min_value=0.0, value=10.0, step=0.5)

    save_to_db = st.button("💾  Save this calculation")


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">Financial Volatility Surface Analyzer</h1>',
            unsafe_allow_html=True)

# ── PRICE CALCULATIONS ────────────────────────────────────────────────────────
if model_choice == "Black-Scholes":
    call_price = black_scholes(S, K, T, r, sigma, dividend_yield, "call")
    put_price  = black_scholes(S, K, T, r, sigma, dividend_yield, "put")
else:
    call_price, call_paths = monte_carlo_simulation(
        S, K, T, r, dividend_yield, sigma, n_paths, n_steps, "call")
    put_price, put_paths   = monte_carlo_simulation(
        S, K, T, r, dividend_yield, sigma, n_paths, n_steps, "put")

# ── PRICE CARDS ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📈 Call Option")
    st.markdown(f"""
    <div style="background:#28A745;padding:25px;border-radius:10px;
                text-align:center;border:2px solid #28A745;">
        <h1 style="margin:0;color:#F8FBF8;">${call_price:.2f}</h1>
        <p style="margin:5px 0 0 0;color:#000B00;">CALL Price</p>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("### 📉 Put Option")
    st.markdown(f"""
    <div style="background:#E53935;padding:25px;border-radius:10px;
                text-align:center;border:2px solid #E53935;">
        <h1 style="margin:0;color:#F8FBF8;">${put_price:.2f}</h1>
        <p style="margin:5px 0 0 0;color:#000B00;">PUT Price</p>
    </div>""", unsafe_allow_html=True)


# ── BLACK-SCHOLES BLOCK ───────────────────────────────────────────────────────
if model_choice == "Black-Scholes":

    # Build shared grids (15x15 keeps annotations readable)
    spot_prices  = np.linspace(spot_min, spot_max, 15)
    volatilities = np.linspace(vol_min,  vol_max,  15)
    spot_grid, vol_grid = np.meshgrid(spot_prices, volatilities)

    call_surface     = np.zeros_like(spot_grid)
    put_surface      = np.zeros_like(spot_grid)
    call_pnl_surface = np.zeros_like(spot_grid)
    put_pnl_surface  = np.zeros_like(spot_grid)

    for i in range(len(volatilities)):
        for j in range(len(spot_prices)):
            cv = black_scholes(spot_grid[i,j], K, T, r, vol_grid[i,j], dividend_yield, "call")
            pv = black_scholes(spot_grid[i,j], K, T, r, vol_grid[i,j], dividend_yield, "put")
            call_surface[i, j]     = cv
            put_surface[i, j]      = pv
            call_pnl_surface[i, j] = cv - call_purchase
            put_pnl_surface[i, j]  = pv - put_purchase

    # 3D Surface
    st.markdown("---")
    st.header("3D Volatility Surface: Spot Price vs Volatility")
    surface_option = st.radio("Show Surface for:", ["Call", "Put"], horizontal=True)
    z_data = call_surface if surface_option == "Call" else put_surface

    fig3d = go.Figure(data=[go.Surface(
        z=z_data, x=spot_prices, y=volatilities,
        colorscale="Viridis",
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.9, roughness=0.4, fresnel=0.2),
        lightposition=dict(x=100, y=100, z=2000),
        hoverinfo="x+y+z", showscale=True,
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
    )])
    fig3d.update_layout(
        title=dict(
            text=f"<b>3D VOLATILITY SURFACE</b><br>{surface_option} | K=${K} | T={T}yr",
            x=0.5, y=0.95, font=dict(size=20, color="white"),
        ),
        scene=dict(
            xaxis_title="<b>SPOT PRICE ($)</b>",
            yaxis_title="<b>VOLATILITY</b>",
            zaxis_title="<b>OPTION PRICE ($)</b>",
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.3)),
            bgcolor="rgb(5,5,15)",
            xaxis=dict(gridcolor="white", backgroundcolor="rgb(20,20,30)"),
            yaxis=dict(gridcolor="white", backgroundcolor="rgb(20,20,30)"),
            zaxis=dict(gridcolor="white", backgroundcolor="rgb(20,20,30)"),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor="black", font=dict(color="white"), height=700,
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # P&L Heatmaps
    st.markdown("---")
    st.header("🟢🔴 P&L Heatmap: Spot Price vs Volatility")
    st.caption(
        f"Green = profit, Red = loss. "
        f"Call purchased at **${call_purchase:.2f}** | Put purchased at **${put_purchase:.2f}**."
    )

    fig_pnl, (ax_call, ax_put) = plt.subplots(1, 2, figsize=(16, 6))
    fig_pnl.patch.set_facecolor("#0a0a0a")
    plot_pnl_heatmap(ax_call, spot_prices, volatilities, call_pnl_surface,
                     "Call Option P&L", call_purchase)
    plot_pnl_heatmap(ax_put,  spot_prices, volatilities, put_pnl_surface,
                     "Put Option P&L",  put_purchase)
    plt.tight_layout()
    st.pyplot(fig_pnl)

    # Contour view
    st.markdown("---")
    st.header("📊 Contour View: Option Price")
    price_surface = call_surface if surface_option == "Call" else put_surface

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    contour  = ax2.contour(spot_grid, vol_grid, price_surface, levels=15,
                            colors="black", alpha=0.6)
    ax2.clabel(contour, inline=True, fontsize=8)
    contourf = ax2.contourf(spot_grid, vol_grid, price_surface, levels=50, cmap="viridis")
    ax2.set_xlabel("Underlying Price ($)")
    ax2.set_ylabel("Volatility")
    ax2.set_title(f"{surface_option} Option Price: Spot vs Volatility (T={T:.1f} years)")
    plt.colorbar(contourf, ax=ax2, label=f"{surface_option} Price ($)")
    st.pyplot(fig2)

    # Save to DB
    if save_to_db:
        calc_id = str(uuid.uuid4())
        save_calculation(
            calc_id, S, K, T, r, sigma, dividend_yield,
            call_price, put_price, call_purchase, put_purchase,
            spot_grid, vol_grid,
            call_surface, put_surface,
            call_pnl_surface, put_pnl_surface,
        )
        st.success(f"✅ Saved — calc ID: `{calc_id[:8]}…`")


# ── MONTE CARLO BLOCK ─────────────────────────────────────────────────────────
else:
    st.header("Monte Carlo Simulation Paths")
    col1, col2 = st.columns(2)

    for col, paths, label, color in [
        (col1, call_paths, "Call", "steelblue"),
        (col2, put_paths,  "Put",  "tomato"),
    ]:
        with col:
            st.subheader(f"{label} Option Paths")
            fig_p, ax_p = plt.subplots(figsize=(10, 6))
            for i in range(len(paths)):
                ax_p.plot(paths[i], alpha=0.3, linewidth=0.5, color=color)
            ax_p.axhline(y=K, color="red", linestyle="--", linewidth=2,
                         label=f"Strike (${K})")
            ax_p.set_xlabel("Trading Days")
            ax_p.set_ylabel("Stock Price ($)")
            ax_p.set_title(f"{label} — {n_paths} Simulations")
            ax_p.legend()
            ax_p.grid(True, alpha=0.3)
            st.pyplot(fig_p)

    st.markdown("---")
    st.header("Final Price Distribution")
    col1, col2 = st.columns(2)

    for col, paths, label, color in [
        (col1, call_paths, "Call", "steelblue"),
        (col2, put_paths,  "Put",  "tomato"),
    ]:
        with col:
            st.subheader(f"{label} Final Prices")
            fig_h, ax_h = plt.subplots(figsize=(10, 4))
            ax_h.hist(paths[:, -1], bins=50, alpha=0.7, color=color, edgecolor="black")
            ax_h.axvline(x=K, color="red", linestyle="--", linewidth=2,
                         label=f"Strike (${K})")
            ax_h.set_xlabel("Final Stock Price ($)")
            ax_h.set_ylabel("Frequency")
            ax_h.set_title(f"Distribution of Final Prices — {label}")
            ax_h.legend()
            st.pyplot(fig_h)


# ── CALCULATION HISTORY ───────────────────────────────────────────────────────
st.markdown("---")
st.header("🗄️ Calculation History")
st.caption("Last 20 saved calculations pulled from the SQLite database.")

history = load_history()
if history.empty:
    st.info("No calculations saved yet. Adjust parameters and click '💾 Save this calculation'.")
else:
    history["timestamp"] = pd.to_datetime(history["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    history.columns = ["Calc ID", "Timestamp", "Spot", "Strike",
                        "Expiry", "Rate", "Vol", "Call $", "Put $"]
    history["Calc ID"] = history["Calc ID"].str[:8] + "…"
    st.dataframe(history, use_container_width=True)
