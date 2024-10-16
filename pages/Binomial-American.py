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
        number_of_steps = st.number_input("Number of Steps (Binomial Model)", min_value=1, max_value=1000, value=100, step=1)
    
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
st.write("### Option Price (American)")
col1, col2 = st.columns(2)
col1.metric(label="American Call Price", value=f"${american_call_price:,.3f}")
col2.metric(label="American Put Price", value=f"${american_put_price:,.3f}")


st.write("### Heatmaps of American Call and Put Prices with Spot Price and Volatility")

# Generate ranges for spot prices and volatilities
spot_range = np.linspace(min_spot, max_spot, 10)  # Increased to 20 for better resolution
volatility_range = np.linspace(min_vol, max_vol, 10)  # Increased to 20

# Create 2D arrays for American call and put prices based on spot prices and volatilities
@st.cache_data  # Cache the heatmap data to optimize performance
def generate_binomial_heatmaps(spot_range, volatility_range, strike_price, time_to_expiry, risk_free_rate, number_of_steps):
    """
    Generates heatmaps data for American call and put options using the Binomial Model.
    
    Returns:
        Tuple of two 2D arrays: (call_prices, put_prices)
    """
    call_prices = np.zeros((len(volatility_range), len(spot_range)))
    put_prices = np.zeros((len(volatility_range), len(spot_range)))
    
    for i, vol in enumerate(volatility_range):
        for j, spot in enumerate(spot_range):
            call_prices[i, j] = binomial_american_option(spot, strike_price, time_to_expiry, risk_free_rate, vol, number_of_steps, option_type="call")
            put_prices[i, j] = binomial_american_option(spot, strike_price, time_to_expiry, risk_free_rate, vol, number_of_steps, option_type="put")
    
    return call_prices, put_prices

call_prices_heatmap, put_prices_heatmap = generate_binomial_heatmaps(
    spot_range, volatility_range, strike_price, time_to_expiry, risk_free_rate, number_of_steps
)

# Plot the heatmap for American Call Prices
fig_call, ax_call = plt.subplots(figsize=(10, 6))
sns.heatmap(call_prices_heatmap, annot=True, fmt=".2f", xticklabels=np.round(spot_range, 2), yticklabels=np.round(volatility_range, 2), cmap="RdYlGn", ax=ax_call)
ax_call.set_title('American Call Option Prices Heatmap (Binomial Model)')
ax_call.set_xlabel('Spot Price')
ax_call.set_ylabel('Volatility')

# Plot the heatmap for American Put Prices
fig_put, ax_put = plt.subplots(figsize=(10, 6))
sns.heatmap(put_prices_heatmap, annot=True, fmt=".2f", xticklabels=np.round(spot_range, 2), yticklabels=np.round(volatility_range, 2), cmap="RdYlGn", ax=ax_put)
ax_put.set_title('American Put Option Prices Heatmap (Binomial Model)')
ax_put.set_xlabel('Spot Price')
ax_put.set_ylabel('Volatility')

# Display Heatmaps
col_heat1, col_heat2 = st.columns(2)
with col_heat1:
    st.pyplot(fig_call)
with col_heat2:
    st.pyplot(fig_put)

import base64

def download_heatmap(heatmap_data, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
    plt.title('Heatmap')
    plt.xlabel('Spot Price')
    plt.ylabel('Volatility')
    plt.tight_layout()

    # Save the figure to a BytesIO object
    from io import BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode to base64
    b64 = base64.b64encode(buf.read()).decode()
    return f'data:image/png;base64,{b64}'

# Add a download button
if st.button("Download Call Price Heatmap"):
    call_heatmap_download = download_heatmap(call_prices_heatmap, "call_prices_heatmap.png")
    st.markdown(f'<a href="{call_heatmap_download}" download="call_prices_heatmap.png">Download Call Prices Heatmap</a>', unsafe_allow_html=True)

if st.button("Download Put Price Heatmap"):
    put_heatmap_download = download_heatmap(put_prices_heatmap, "put_prices_heatmap.png")
    st.markdown(f'<a href="{put_heatmap_download}" download="put_prices_heatmap.png">Download Put Prices Heatmap</a>', unsafe_allow_html=True)
    
st.markdown("---")
st.markdown("### Developed by Kafui Avevor")
st.markdown("### [LinkedIn](https://www.linkedin.com/in/kafui-avevor/) | [GitHub](https://github.com/kafuiavevor)")

