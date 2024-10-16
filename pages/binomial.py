import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

st.title("Binomial Model for American Options")
st.markdown("### By Kafui Avevor")

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

# Binomial Model for American Options
def binomial_american_option(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, number_of_steps, option_type='call'):
    """
    Binomial model for pricing American options.
    """
    # Calculate the time step
    dt = time_to_expiry / number_of_steps
    # Up and down factors
    u = np.exp(volatility * np.sqrt(dt))
    d = 1 / u
    # Risk-neutral probability
    p = (np.exp(risk_free_rate * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity
    asset_prices = np.array([spot_price * (u ** i) * (d ** (number_of_steps - i)) for i in range(number_of_steps + 1)])
    
    # Initialize option values at maturity
    if option_type == 'call':
        option_values = np.maximum(asset_prices - strike_price, 0)
    elif option_type == 'put':
        option_values = np.maximum(strike_price - asset_prices, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    # Step back through the tree
    for j in range(number_of_steps - 1, -1, -1):
        for i in range(j + 1):
            # Calculate asset price at node (j, i)
            S = spot_price * (u ** i) * (d ** (j - i))
            # Continuation value
            hold_value = np.exp(-risk_free_rate * dt) * (p * option_values[i + 1] + (1 - p) * option_values[i])
            # Early exercise value
            if option_type == 'call':
                exercise_value = max(S - strike_price, 0)
            else:  # put
                exercise_value = max(strike_price - S, 0)
            # Option value at node (j, i)
            option_values[i] = max(hold_value, exercise_value)
    
    return option_values[0]

# Calculate American Binomial Option Prices
american_call_price = binomial_american_option(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, number_of_steps, option_type='call')
american_put_price = binomial_american_option(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, number_of_steps, option_type='put')

# Display Option Prices (American Binomial)
st.write("### Option Price (American Binomial Model)")
col1, col2 = st.columns(2)
col1.metric(label="American Call Price", value=f"${american_call_price:,.3f}")
col2.metric(label="American Put Price", value=f"${american_put_price:,.3f}")
