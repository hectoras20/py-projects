#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 23:36:54 2025

@author: hectorastudillo
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Descargar datos ---
ticker = "LIVEPOLC-1.MX"
df = yf.download(ticker, start="2021-01-01")

# --- Calcular RSI manualmente ---
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)   # solo valores positivos
    loss = -delta.clip(upper=0)  # solo valores negativos (en positivo)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df["RSI"] = rsi(df["Close"], 14)

# --- Calcular OBV manualmente ---
obv = [0]
for i in range(1, len(df)):
    close_now = float(df["Close"].iloc[i])
    close_prev = float(df["Close"].iloc[i-1])
    vol_now = float(df["Volume"].iloc[i])

    if close_now > close_prev:
        obv.append(obv[-1] + vol_now)
    elif close_now < close_prev:
        obv.append(obv[-1] - vol_now)
    else:
        obv.append(obv[-1])

df["OBV"] = obv

# --- Graficar ---
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                        gridspec_kw={'height_ratios': [3, 1, 1]})

# 1. Precio
axs[0].plot(df.index, df["Close"], color="black", label="Cierre")
axs[0].set_title("LIVEPOLC-1.MX (Liverpool) - Precio, RSI y OBV")
axs[0].legend()
axs[0].grid(alpha=0.3)

# 2. RSI
axs[1].plot(df.index, df["RSI"], color="purple", label="RSI 14")
axs[1].axhline(70, color="red", linestyle="--", alpha=0.6)
axs[1].axhline(30, color="green", linestyle="--", alpha=0.6)
axs[1].legend()
axs[1].grid(alpha=0.3)

# 3. OBV
axs[2].plot(df.index, df["OBV"], color="blue", label="OBV")
axs[2].legend()
axs[2].grid(alpha=0.3)

axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

