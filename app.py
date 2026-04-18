import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import plotly.graph_objects as go


JAPANESE_PALETTE = {
    "Background": "#000B00",
    "Accent": "#180614",     
    "Primary": "#250D00",    
    "Header": "#000B00",     
    "Text": "#F3F3F2"        
}

st.set_page_config(
    page_title="Financial Volatility Surface Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply the color palette
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {JAPANESE_PALETTE['Background']};
        color: {JAPANESE_PALETTE['Text']};
    }}
    .main-header {{
        color: {JAPANESE_PALETTE['Text']};
        border-bottom: 2px solid {JAPANESE_PALETTE['Header']};
        padding-bottom: 10px;
    }}
    .metric-card {{
        background-color: {JAPANESE_PALETTE['Primary']};
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid {JAPANESE_PALETTE['Accent']};
    }}
    </style>
""", unsafe_allow_html=True)

# FINANCIAL MODEL IMPLEMENTATIONS 
def black_scholes(S, K, T, r, sigma,s_sigma=0.0, option_type='call'):
    d1 = (np.log(S / K) + (r - s_sigma + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be 'call' or 'put'")
    
    return price

def monte_carlo_simulation(S, K, T, r, s_sigma, sigma, n_paths=1000, n_steps=252, option_type='call'):
    paths = np.zeros((n_paths, n_steps+1))
    paths[:,0] = S

    z = np.random.standard_normal((n_paths,n_steps))
    h = T/n_steps
    for t in range(1,n_steps+1):
        paths[:,t] = paths[:,t-1]*(np.exp((r-s_sigma-(0.5*(sigma**2)))*h + (sigma * np.sqrt(h) * z[:,t-1])))

    if option_type == 'call':
        payoffs = np.maximum(paths[:,-1]-K, 0)
    else:
        payoffs = np.maximum(K-paths[:,-1],0)

    price = np.exp(-r * T)*np.mean(payoffs)
    return price, paths

# UI layout using streamlit 
with st.sidebar:
    st.header("Parameters and control panel")
    
    st.subheader("Model Configuration")
    model_choice = st.selectbox(
        "Pricing Model",
        ["Black-Scholes", "Monte Carlo"],
        help="Choose your pricing model"
    )
    
    # Essential Parameters 
    st.subheader("Base Scenario")
    S = st.slider("Spot Price", 50.0, 150.0, 100.0, 5.0)
    K = st.slider("Strike Price", 50.0, 150.0, 100.0, 5.0)
    T = st.slider("Time to Expiry (Years)", 0.1, 2.0, 1.0, 0.1)
    r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.5) / 100
    sigma = st.slider("Volatility (%)", 10.0, 100.0, 30.0, 5.0) / 100
    dividend_yield = st.slider("Dividend Yield (%)", 0.0, 10.0, 0.0, 0.5) / 100
    
    if model_choice == "Monte Carlo":
        st.subheader("Advanced Settings")
        n_paths = st.slider("Number of Paths", 100, 10000, 1000, 100, help="More paths = more accuracy but slower computation")
        n_steps = st.slider("Number of Steps", 10, 500, 252, 10)
    else:
        # Shock ranges for Black-Scholes heatmap
        st.subheader("Heatmap Shock Ranges")
        vol_min = st.slider("Min Volatility (%)", 10.0, 50.0, 15.0, 5.0) / 100
        vol_max = st.slider("Max Volatility (%)", 30.0, 100.0, 60.0, 5.0) / 100
        spot_min = st.slider("Min Spot Price", S * 0.5, S, S * 0.7, 5.0)
        spot_max = st.slider("Max Spot Price", S, S * 1.5, S * 1.3, 5.0)

        time_min = st.slider("Max Time ", 1.0, 2.0, 1.0 ,3/12)
        time_max = st.slider("Max Time ", 2.0, 10.0, 3.0 ,3/12)






# MAIN DASHBOARD
st.markdown('<h1 class="main-header">Financial Volatility Surface Analyzer</h1>', unsafe_allow_html=True)

# Calculate BOTH call and put prices regardless of model choice
if model_choice == "Black-Scholes":
    call_price = black_scholes(S, K, T, r, sigma,dividend_yield, "call")
    put_price = black_scholes(S, K, T, r, sigma,dividend_yield, "put")
else:
    # Monte Carlo pricing for both
    call_price, call_paths = monte_carlo_simulation(S, K, T, r, dividend_yield, sigma, n_paths, n_steps, "call")
    put_price, put_paths = monte_carlo_simulation(S, K, T, r, dividend_yield, sigma, n_paths, n_steps, "put")

# Display BOTH Option Prices (always show both)
col1, col2 = st.columns(2)

with col1:
    # CALL OPTION CARD
    st.markdown("### 📈 Call Option")
    st.markdown(f"""
    <div style="background: #28A745; padding: 25px; border-radius: 10px; text-align: center; border: 2px solid #28A745;">
        <h1 style="margin: 0; color: #F8FBF8;">${call_price:.2f}</h1>
        <p style="margin: 5px 0 0 0; color: #000B00;"> CALL Price$</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # PUT OPTION CARD
    st.markdown("### 📉 Put Option")
    st.markdown(f"""
    <div style="background: #E53935; padding: 25px; border-radius: 10px; text-align: center; border: 2px solid #E53935;">
        <h1 style="margin: 0; color: #F8FBF8;">${put_price:.2f}</h1>
        <p style="margin: 5px 0 0 0; color: #000B00;"> PUT Price$</p>
    </div>
    """, unsafe_allow_html=True)





# 3D SURFACE PLOT - SPOT PRICE vs VOLATILITY
# BEAUTIFUL 3D SURFACE PLOT WITH PLOTLY - SPOT PRICE vs VOLATILITY
if model_choice == "Black-Scholes":
    st.markdown("---")
    st.header("3D Volatility Surface: Spot Price vs Volatility")
    
    # Let user choose which option type to show
    surface_option = st.radio("Show Surface for:", ["Call", "Put"], horizontal=True)
    
    # Generate grid for 3D surface - SPOT vs VOLATILITY
    spot_prices = np.linspace(spot_min, spot_max, 25)
    volatilities = np.linspace(vol_min, vol_max, 25)
    spot_grid, vol_grid = np.meshgrid(spot_prices, volatilities)
    
    # Calculate option prices for the grid (TIME IS NOW FIXED)
    price_surface = np.zeros_like(spot_grid)
    for i in range(len(volatilities)):
        for j in range(len(spot_prices)):
            price_surface[i, j] = black_scholes(spot_grid[i, j], K, T, r, vol_grid[i, j], dividend_yield, surface_option.lower())
    
    # CREATE BEAUTIFUL PLOTLY 3D SURFACE
    fig = go.Figure(data=[go.Surface(
        z=price_surface,
        x=spot_prices,
        y=volatilities,
        colorscale='Viridis',
        lighting=dict(
            ambient=0.4,
            diffuse=0.8,
            specular=0.9,
            roughness=0.4,
            fresnel=0.2
        ),
        lightposition=dict(x=100, y=100, z=2000),
        hoverinfo='x+y+z',
        showscale=True,
        contours=dict(
            z=dict(show=True, usecolormap=True, project_z=True)
        )
    )])
    
    fig.update_layout(
        title=dict(
            text=f'<b>3D VOLATILITY SURFACE</b><br>{surface_option} Options | K=${K} | T={T}yr',
            x=0.5,
            y=0.95,
            font=dict(size=20, color='white', family='Arial')
        ),
        scene=dict(
            xaxis_title='<b>SPOT PRICE ($)</b>',
            yaxis_title='<b>VOLATILITY</b>', 
            zaxis_title='<b>OPTION PRICE ($)</b>',
            camera=dict(
                eye=dict(x=1.7, y=1.7, z=1.3)  # DRAMATIC 3D ANGLE
            ),
            bgcolor='rgb(5,5,15)',
            xaxis=dict(
                gridcolor='white',
                gridwidth=2,
                backgroundcolor='rgb(20,20,30)'
            ),
            yaxis=dict(
                gridcolor='white',
                gridwidth=2, 
                backgroundcolor='rgb(20,20,30)'
            ),
            zaxis=dict(
                gridcolor='white',
                gridwidth=2,
                backgroundcolor='rgb(20,20,30)'
            )
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor='black',
        font=dict(color='white'),
        height=700  # Larger for better 3D experience
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 2D CONTOUR PLOT - SPOT vs VOLATILITY
    st.markdown("---")
    st.header("📊 Contour View: Spot Price vs Volatility")
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    contour = ax2.contour(spot_grid, vol_grid, price_surface, levels=15, colors='black', alpha=0.6)
    ax2.clabel(contour, inline=True, fontsize=8)
    contourf = ax2.contourf(spot_grid, vol_grid, price_surface, levels=50, cmap='viridis')
    
    ax2.set_xlabel('Underlying Price ($)')
    ax2.set_ylabel('Volatility')
    ax2.set_title(f'{surface_option} Option Price: Spot vs Volatility (T={T:.1f} years)')
    plt.colorbar(contourf, ax=ax2, label=f'{surface_option} Price ($)')
    
    st.pyplot(fig2)
    



# MONTE CARLO VISUALIZATION - USING THE SIMPLE APPROACH
else:
    st.header("Monte Carlo Simulation Paths")
    
    # Create a two-column layout for call and put paths
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Call Option Paths")
        
        # Create call paths visualization - SIMPLE like your example
        fig_call, ax_call = plt.subplots(figsize=(10, 6))
        
        # Plot all simulation paths with transparent lines
        for i in range(len(call_paths)):
            ax_call.plot(call_paths[i])
        
        # Add strike line
        ax_call.axhline(y=K, color='red', linestyle='--', linewidth=2, label=f'Strike Price (${K})')
        
        ax_call.set_xlabel('Trading Days')
        ax_call.set_ylabel('Stock Price ($)')
        ax_call.set_title(f'Call Option - {n_paths} Simulations')
        ax_call.legend()
        ax_call.grid(True, alpha=0.3)
        
        st.pyplot(fig_call)
    
    with col2:
        st.subheader("Put Option Paths")
        
        # Create put paths visualization - SIMPLE like your example
        fig_put, ax_put = plt.subplots(figsize=(10, 6))
        
        # Plot all simulation paths with transparent lines
        for i in range(len(put_paths)):
            ax_put.plot(put_paths[i])
        
        # Add strike line
        ax_put.axhline(y=K, color='red', linestyle='--', linewidth=2, label=f'Strike Price (${K})')
        
        ax_put.set_xlabel('Trading Days')
        ax_put.set_ylabel('Stock Price ($)')
        ax_put.set_title(f'Put Option - {n_paths} Simulations')
        ax_put.legend()
        ax_put.grid(True, alpha=0.3)
        
        st.pyplot(fig_put)
    
    # HISTOGRAM OF FINAL PRICES (exactly like your example)
    st.markdown("---")
    st.header("Final Price Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Call Option Final Prices")
        fig_hist_call, ax_hist_call = plt.subplots(figsize=(10, 4))
        
        # Get final prices from all call simulations
        final_prices_call = call_paths[:, -1]
        
        # Create histogram - exactly like your example
        ax_hist_call.hist(final_prices_call, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax_hist_call.axvline(x=K, color='red', linestyle='--', linewidth=2, label=f'Strike Price (${K})')
        
        ax_hist_call.set_xlabel('Final Stock Price ($)')
        ax_hist_call.set_ylabel('Frequency')
        ax_hist_call.set_title('Distribution of Final Prices - Call')
        ax_hist_call.legend()
        
        st.pyplot(fig_hist_call)
    
    with col2:
        st.subheader("Put Option Final Prices")
        fig_hist_put, ax_hist_put = plt.subplots(figsize=(10, 4))
        
        # Get final prices from all put simulations
        final_prices_put = put_paths[:, -1]
        
        # Create histogram - exactly like your example
        ax_hist_put.hist(final_prices_put, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax_hist_put.axvline(x=K, color='red', linestyle='--', linewidth=2, label=f'Strike Price (${K})')
        
        ax_hist_put.set_xlabel('Final Stock Price ($)')
        ax_hist_put.set_ylabel('Frequency')
        ax_hist_put.set_title('Distribution of Final Prices - Put')
        ax_hist_put.legend()
        
        st.pyplot(fig_hist_put)

