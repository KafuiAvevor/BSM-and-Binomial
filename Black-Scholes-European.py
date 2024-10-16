import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import math

st.title("Black Scholes Pricing Model")
with st.sidebar:
    st.title("Black Scholes Pricing Model")
    st.subheader("By Kafui Avevor")
    st.write("### Input Data")
    col1, col2 = st.columns(2)
    spot_price = col1.number_input("Spot Price", min_value=0.00, value=50.00)
    strike_price = col1.number_input("Strike Price", min_value=0.00, value=55.00)
    risk_free_rate = col1.number_input("Risk Free Rate", min_value=0.00, value=0.05)
    time_to_expiry = col2.number_input("Time to Expiry (in years)", min_value=0.00, value=1.00)
    volatility = col2.number_input("Volatility", min_value=0.00, value=0.2)
    number_of_steps = col2.number_input("number_of_steps", min_value=0, value=100)


# Black Scholes Model
def black_scholes(spot_price, strike_price, risk_free_rate, time_to_expiry, volatility, option_type="call"):
    """
  This function calculates the Black-Scholes option price.
 
  Args:
      spot_price (float): The current price of the underlying asset.
      strike_price (float): The strike price of the option.
      risk_free_rate (float): The risk-free interest rate.
      time_to_expiry (float): The time to expiry of the option in years.
      volatility (float): The implied volatility of the underlying asset.
      option_type (str, optional): The type of option ("call" or "put"). Defaults to "call".
    Returns:
      float: The Black-Scholes option price.
  """
    d1 = (np.log(spot_price / strike_price) + (risk_free_rate + volatility**2 / 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    if option_type == "call":
        price = spot_price * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
    elif option_type == "put":
        price = strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Please use 'call' or 'put'.")
    return price 

# Display the option price
call_price = black_scholes(spot_price, strike_price, risk_free_rate, time_to_expiry, volatility)
put_price = black_scholes(spot_price, strike_price, risk_free_rate, time_to_expiry, volatility, option_type="put")

st.write("### Option Price (European")
col1, col2 = st.columns(2)
col1.metric(label="European Call Price", value=f"${call_price:,.3f}")
col2.metric(label="European Put Price", value=f"${put_price:,.3f}")


with st.sidebar:
    st.write("### Heatmap Parameters")
    min_vol = st.slider("Min Volatility", 0.00, 1.00, volatility*0.5)
    max_vol = st.slider("Max Volatility",0.00, 1.00, volatility*1.5)
    min_spot = st.number_input("Min Spot Price",0.00, 1000.00, spot_price*0.5)
    max_spot = st.number_input("Max Spot Price",0.00, 1000.00, spot_price*1.5)
# Generate the heatmap data (for Call and Put Prices with different Spot Prices and Volatilities)
st.write("### Heatmaps of European Call and Put Prices with Volatility and Spot Price")

spot_range = np.linspace(min_spot, max_spot, 10)  # 10 different spot prices
volatility_range = np.linspace(min_vol, max_vol , 10)  # 10 different volatilities


# Create 2D arrays for call and put prices based on spot prices and volatilities
call_prices = np.zeros((len(volatility_range), len(spot_range)))
put_prices = np.zeros((len(volatility_range), len(spot_range)))

for i, vol in enumerate(volatility_range):
    for j, spot in enumerate(spot_range):
        call_prices[i, j] = black_scholes(spot, strike_price, risk_free_rate, time_to_expiry, vol)
        put_prices[i, j] = black_scholes(spot, strike_price, risk_free_rate, time_to_expiry, vol, option_type="put")

fig, (ax_call, ax_put) = plt.subplots(1, 2, figsize=(16, 6))

# Plot the heatmap for Call Prices on the first subplot
sns.heatmap(call_prices, annot=True, fmt=".2f", xticklabels=np.round(spot_range, 2),
            yticklabels=np.round(volatility_range, 2), cmap="RdYlGn", ax=ax_call)
ax_call.set_title('Call Option Prices Heatmap')
ax_call.set_xlabel('Spot Price')
ax_call.set_ylabel('Volatility')

# Plot the heatmap for Put Prices on the second subplot
sns.heatmap(put_prices, annot=True, fmt=".2f", xticklabels=np.round(spot_range, 2),
            yticklabels=np.round(volatility_range, 2), cmap="RdYlGn", ax=ax_put)
ax_put.set_title('Put Option Prices Heatmap')
ax_put.set_xlabel('Spot Price')
ax_put.set_ylabel('Volatility')

# Adjust layout and display heatmaps
plt.tight_layout()
st.pyplot(fig)


#Calculate the Greeks
d1 = (np.log(spot_price / strike_price) + (risk_free_rate + volatility**2 / 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
d2 = d1 - volatility * np.sqrt(time_to_expiry)
delta_call = norm.cdf(d1)
delta_put = norm.cdf(d1)-1
gamma = norm.pdf(d1) / (spot_price * volatility *(np.sqrt(time_to_expiry)))
theta_call = -((spot_price*norm.pdf(d1)*volatility)/(2*np.sqrt(time_to_expiry))) - risk_free_rate *strike_price*np.exp(-risk_free_rate*time_to_expiry)*norm.cdf(d2) 
theta_put = -((spot_price*norm.pdf(d1)*volatility)/(2*np.sqrt(time_to_expiry))) + risk_free_rate *strike_price*np.exp(-risk_free_rate*time_to_expiry)*norm.cdf(-d2)
rho_call = 0.01*strike_price*time_to_expiry* np.exp(-risk_free_rate*time_to_expiry) * norm.cdf(d2)
rho_put = -0.01*(strike_price*time_to_expiry* np.exp(-risk_free_rate*time_to_expiry) * norm.cdf(-d2))
vega = 0.01*spot_price*norm.pdf(d1)*np.sqrt(time_to_expiry)
st.write("### Greeks")

col1, col2 = st.columns(2)
col1.metric(label="Call Delta", value=f"{delta_call:,.3f}")
col2.metric(label="Put Delta", value=f"{delta_put:,.3f}")

col1, col2 = st.columns(2)
col1.metric(label="Gamma", value=f"{gamma:,.3f}")
col2.metric(label="Vega", value=f"{vega:,.3f}")

col1, col2 = st.columns(2)
col1.metric(label="Call Theta", value=f"{theta_call:,.3f}")
col2.metric(label="Put Theta", value=f"{theta_put:,.3f}")

col1, col2 = st.columns(2)
col1.metric(label="Call Rho", value=f"{rho_call:,.3f}")
col2.metric(label="Put Rho", value=f"{rho_put:,.3f}")

