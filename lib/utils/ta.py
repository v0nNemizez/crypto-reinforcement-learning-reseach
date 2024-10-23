def calculate_rsi(prices, window=50):  # Øker vinduet til 50 for minuttsdata

    delta = prices.diff().fillna(0)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi_normalized = rsi / 100
    rsi_normalized.fillna(0.5, inplace=True)

    return rsi_normalized
        
def calculate_sma(prices, current_step, window=50):
        # SMA (Simple Moving Average) beregning
        if current_step < window:
            return 0.5  # Hvis det er for tidlig i datasettet, returner en nøytral verdi

        sma = prices.rolling(window=window).mean()
        current_price = prices.iloc[current_step]
        max_price = prices.max()
        return (current_price / sma.iloc[current_step]) / max_price