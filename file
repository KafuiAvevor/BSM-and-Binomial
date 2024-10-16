import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import math

st.set_page_config(page_title="Option Pricing Models", layout="wide")

st.title("Option Pricing Models")
st.markdown("### Comparing Black-Scholes-Merton (BSM) and Binomial Models for Option Pricing")
st.markdown("#### By Kafui Avevor")

# Sidebar Inputs
with st.sidebar:
    st.header("Input Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        spot_price = st.number_input("Spot Price", min_value=0.00, value=50.00, step=0.1)
        strike_price = st.number_input("Strike Price", min_value=0.00, value=55.00, step=0.1)
        risk_free_rate = st.number_input("Risk-Free Rate (e.g., 0.05 for 5%)", min_value=0.00, value=0.05, step=0.001)
    
    with col2:
        time_to_expiry = st.number_input("Time to Expiry (in years)", min_value=0.00, value=1.00, step=0.01)
        volatility = st.number_input("Volatility (e.g., 0.2 for 20%)", min_value=0.00, value=0.2, step=0.01)
        number_of_steps = st.number_input("Number of Steps (Binomial Model)", min_value=10, max_value=1000, value=100, step=10)
    
    st.markdown("---")
    st.header("Heatmap Parameters")
    min_vol = st.slider("Min Volatility", 0.00, 1.00, float(volatility)*0.5, step=0.01)
    max_vol = st.slider("Max Volatility", 0.00, 1.00, float(volatility)*1.5, step=0.01)
    min_spot = st.number_input("Min Spot Price for Heatmap", min_value=0.00, max_value=10000.00, value=float(spot_price)*0.5, step=0.1)
    max_spot = st.number_input("Max Spot Price for Heatmap", min_value=0.00, max_value=10000.00, value=float(spot_price)*1.5, step=0.1)

# Black-Scholes-Merton (BSM) Model Function
def black_scholes(spot_price, strike_price, risk_free_rate, time_to_expiry, volatility, option_type="call"):
    """
    Calculates the Black-Scholes option price.
    """
    if time_to_expiry <= 0 or volatility <= 0 or spot_price <= 0 or strike_price <= 0:
        return np.nan  # Return Not a Number for invalid inputs

    d1 = (np.log(spot_price / strike_price) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    
    if option_type == "call":
        price = spot_price * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
    elif option_type == "put":
        price = strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Please use 'call' or 'put'.")
    return price 

# Binomial Model for American Options
def binomial_american_option(S0, K, T, r, sigma, N, option_type='call'):
    """
    Binomial model for pricing American options.
    """
    # Calculate the time step
    dt = T / N
    # Up and down factors
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    # Risk-neutral probability
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity
    asset_prices = np.array([S0 * (u ** i) * (d ** (N - i)) for i in range(N + 1)])
    
    # Initialize option values at maturity
    if option_type == 'call':
        option_values = np.maximum(asset_prices - K, 0)
    elif option_type == 'put':
        option_values = np.maximum(K - asset_prices, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    # Step back through the tree
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            # Calculate asset price at node (j, i)
            S = S0 * (u ** i) * (d ** (j - i))
            # Continuation value
            hold_value = np.exp(-r * dt) * (p * option_values[i + 1] + (1 - p) * option_values[i])
            # Early exercise value
            if option_type == 'call':
                exercise_value = max(S - K, 0)
            else:  # put
                exercise_value = max(K - S, 0)
            # Option value at node (j, i)
            option_values[i] = max(hold_value, exercise_value)
    
    return option_values[0]

# Calculate BSM Prices
call_price_bsm = black_scholes(spot_price, strike_price, risk_free_rate, time_to_expiry, volatility, option_type='call')
put_price_bsm = black_scholes(spot_price, strike_price, risk_free_rate, time_to_expiry, volatility, option_type='put')

# Calculate Binomial American Option Prices
american_call_price = binomial_american_option(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, number_of_steps, option_type='call')
american_put_price = binomial_american_option(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, number_of_steps, option_type='put')

# Display Option Prices
st.subheader("Option Prices Comparison")
col1, col2, col3, col4 = st.columns(4)

col1.metric(label="BSM Call Price", value=f"${call_price_bsm:,.3f}")
col2.metric(label="Binomial American Call Price", value=f"${american_call_price:,.3f}")
col3.metric(label="BSM Put Price", value=f"${put_price_bsm:,.3f}")
col4.metric(label="Binomial American Put Price", value=f"${american_put_price:,.3f}")

# Greeks Calculation (BSM)
def calculate_greeks(S, K, T, r, sigma):
    """
    Calculates the Greeks for European options using BSM.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {'Delta': np.nan, 'Gamma': np.nan, 'Theta': np.nan, 'Vega': np.nan, 'Rho': np.nan}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta_call = norm.cdf(d1)
    delta_put = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta_call = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2)
    theta_put = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change
    rho_call = 0.01 * K * T * np.exp(-r * T) * norm.cdf(d2)
    rho_put = -0.01 * K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    return {
        'Delta Call': delta_call,
        'Delta Put': delta_put,
        'Gamma': gamma,
        'Theta Call': theta_call,
        'Theta Put': theta_put,
        'Vega': vega,
        'Rho Call': rho_call,
        'Rho Put': rho_put
    }

greeks = calculate_greeks(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility)

# Display Greeks
st.subheader("Option Greeks (BSM Model)")
col_g1, col_g2, col_g3, col_g4 = st.columns(4)

col_g1.metric(label="Delta Call", value=f"{greeks['Delta Call']:.3f}")
col_g2.metric(label="Delta Put", value=f"{greeks['Delta Put']:.3f}")
col_g3.metric(label="Gamma", value=f"{greeks['Gamma']:.3f}")
col_g4.metric(label="Vega", value=f"{greeks['Vega']:.3f}")

col_g1, col_g2 = st.columns(2)
col_g1.metric(label="Theta Call", value=f"{greeks['Theta Call']:.3f}")
col_g2.metric(label="Theta Put", value=f"{greeks['Theta Put']:.3f}")

col_g1, col_g2 = st.columns(2)
col_g1.metric(label="Rho Call", value=f"{greeks['Rho Call']:.3f}")
col_g2.metric(label="Rho Put", value=f"{greeks['Rho Put']:.3f}")

# Heatmap Data Generation
st.subheader("Heatmaps of Call and Put Prices with Spot Price and Volatility")

# Generate ranges for spot prices and volatilities
spot_range = np.linspace(min_spot, max_spot, 20)  # Increased to 20 for better resolution
volatility_range = np.linspace(min_vol, max_vol, 20)

# Create meshgrid for heatmap
Spot, Vol = np.meshgrid(spot_range, volatility_range)

# Calculate Call and Put Prices using BSM for heatmap
call_prices_heatmap = np.vectorize(lambda S, V: black_scholes(S, strike_price, risk_free_rate, time_to_expiry, V, 'call'))(Spot, Vol)
put_prices_heatmap = np.vectorize(lambda S, V: black_scholes(S, strike_price, risk_free_rate, time_to_expiry, V, 'put'))(Spot, Vol)

# Plot Heatmap for Call Prices
fig_call, ax_call = plt.subplots(figsize=(10, 6))
sns.heatmap(call_prices_heatmap, annot=True, fmt=".2f", xticklabels=np.round(spot_range, 2), yticklabels=np.round(volatility_range, 2), cmap="RdYlGn", ax=ax_call)
ax_call.set_title('Call Option Prices Heatmap (BSM)')
ax_call.set_xlabel('Spot Price')
ax_call.set_ylabel('Volatility')

# Plot Heatmap for Put Prices
fig_put, ax_put = plt.subplots(figsize=(10, 6))
sns.heatmap(put_prices_heatmap, annot=True, fmt=".2f", xticklabels=np.round(spot_range, 2), yticklabels=np.round(volatility_range, 2), cmap="RdYlGn", ax=ax_put)
ax_put.set_title('Put Option Prices Heatmap (BSM)')
ax_put.set_xlabel('Spot Price')
ax_put.set_ylabel('Volatility')

# Display Heatmaps
col_heat1, col_heat2 = st.columns(2)
with col_heat1:
    st.pyplot(fig_call)
with col_heat2:
    st.pyplot(fig_put)

# Comparison of BSM and Binomial Model Prices
st.subheader("Comparison between BSM and Binomial Models")

comparison_data = pd.DataFrame({
    'Model': ['BSM Call', 'Binomial American Call', 'BSM Put', 'Binomial American Put'],
    'Price': [call_price_bsm, american_call_price, put_price_bsm, american_put_price]
})

fig_comparison, ax_comparison = plt.subplots(figsize=(8, 4))
sns.barplot(x='Model', y='Price', data=comparison_data, palette='viridis', ax=ax_comparison)
ax_comparison.set_title('Option Prices: BSM vs. Binomial Model')
ax_comparison.set_ylabel('Price')
for index, row in comparison_data.iterrows():
    ax_comparison.text(index, row.Price + 0.5, f"${row.Price:.2f}", ha='center')
st.pyplot(fig_comparison)

# Optional: Display Data Table
with st.expander("Show Data Table"):
    st.dataframe(comparison_data)

# Footer
st.markdown("---")
st.markdown("### Developed by Kafui Avevor")
st.markdown("### [LinkedIn](https://www.linkedin.com/in/kafui-avevor/) | [GitHub](https://github.com/kafuiavevor)")
