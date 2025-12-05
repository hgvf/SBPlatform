# ----------------- 進場策略 ----------------- #

def signal_ma_golden_cross(data, fast_period=5, slow_period=20):
    """均線黃金交叉：快線向上突破慢線"""

    fast_ma = f'SMA_{fast_period}' if f'SMA_{fast_period}' in data.lines.getlinealiases() else f'EMA_{fast_period}'
    slow_ma = f'SMA_{slow_period}' if f'SMA_{slow_period}' in data.lines.getlinealiases() else f'EMA_{slow_period}'
    
    fast_line = getattr(data, fast_ma)
    slow_line = getattr(data, slow_ma)
    
    # 當前快線 > 慢線 且 前一根快線 <= 慢線
    return fast_line[0] > slow_line[0] and fast_line[-1] <= slow_line[-1]

def signal_ma_bullish(data, fast_period=5, slow_period=20):
    """均線多頭排列：快線在慢線之上"""

    fast_ma = f'SMA_{fast_period}' if f'SMA_{fast_period}' in data.lines.getlinealiases() else f'EMA_{fast_period}'
    slow_ma = f'SMA_{slow_period}' if f'SMA_{slow_period}' in data.lines.getlinealiases() else f'EMA_{slow_period}'
    
    fast_line = getattr(data, fast_ma)
    slow_line = getattr(data, slow_ma)
    
    return fast_line[0] > slow_line[0]

def signal_price_above_ma(data, ma_period=20, ma_type='SMA'):
    """價格突破均線"""
    ma_name = f'{ma_type}_{ma_period}'
    ma_line = getattr(data, ma_name)
    
    return data.close[0] > ma_line[0] and data.close[-1] <= ma_line[-1]

def signal_ma_slope_up(data, ma_period=20, ma_type='SMA', lookback=3):
    """均線向上傾斜"""
    ma_name = f'{ma_type}_{ma_period}'
    ma_line = getattr(data, ma_name)
    
    # 檢查最近N根K線均線持續上升
    return all(ma_line[-i] < ma_line[-i+1] for i in range(lookback, 0, -1))

def signal_rsi_oversold_cross(data, rsi_period=14, threshold=30):
    """RSI超賣後向上突破"""
    rsi_name = f'RSI_{rsi_period}'
    rsi = getattr(data, rsi_name)
    
    # RSI從低於threshold向上突破
    return rsi[0] > threshold and rsi[-1] <= threshold

def signal_rsi_bullish(data, rsi_period=14, lower=40, upper=70):
    """RSI處於多頭區間"""
    rsi_name = f'RSI_{rsi_period}'
    rsi = getattr(data, rsi_name)
    
    return lower < rsi[0] < upper

def signal_macd_golden_cross(data):
    """MACD黃金交叉：MACD線向上突破信號線"""
    macd = data.MACD
    signal = data.MACD_signal
    
    return macd[0] > signal[0] and macd[-1] <= signal[-1]

def signal_macd_bullish(data):
    """MACD多頭：MACD線在信號線之上"""
    return data.MACD[0] > data.MACD_signal[0]

def signal_macd_hist_positive(data):
    """MACD柱狀圖轉正"""
    return data.MACD_hist[0] > 0 and data.MACD_hist[-1] <= 0

def signal_kd_golden_cross(data):
    """KD黃金交叉：K線向上突破D線"""
    k = data.KD_K
    d = data.KD_D
    
    return k[0] > d[0] and k[-1] <= d[-1]

def signal_kd_oversold_cross(data, threshold=20):
    """KD超賣區黃金交叉"""
    k = data.KD_K
    d = data.KD_D
    
    # K線向上突破D線且兩者都在超賣區
    return (k[0] > d[0] and k[-1] <= d[-1] and 
            k[0] < threshold and d[0] < threshold)

def signal_kd_bullish(data):
    """KD多頭：K線在D線之上"""
    return data.KD_K[0] > data.KD_D[0]

def signal_bb_lower_bounce(data):
    """布林通道下軌反彈"""
    close = data.close
    bb_lower = data.BB_lower
    
    # 價格從下軌反彈
    return close[-1] <= bb_lower[-1] and close[0] > bb_lower[0]

def signal_bb_squeeze_break(data):
    """布林通道收縮後突破"""
    bandwidth = data.BB_bandwidth
    close = data.close
    bb_middle = data.BB_middle
    
    # 通道寬度收縮且價格突破中軌
    return (bandwidth[0] < bandwidth[-5] and 
            close[0] > bb_middle[0] and close[-1] <= bb_middle[-1])

def signal_bb_percent_b_bullish(data, threshold=0.5):
    """布林%B指標多頭"""
    return data.BB_percent_b[0] > threshold

def signal_cci_oversold_cross(data, threshold=-100):
    """CCI超賣後向上突破"""
    cci = data.CCI_20
    
    return cci[0] > threshold and cci[-1] <= threshold

def signal_cci_bullish(data):
    """CCI多頭區間"""
    return data.CCI_20[0] > 0

def signal_willr_oversold_cross(data, threshold=-80):
    """威廉指標超賣後向上突破"""
    willr = data.WILLR_14
    
    return willr[0] > threshold and willr[-1] <= threshold

def signal_mfi_oversold_cross(data, threshold=20):
    """資金流量指標超賣後向上突破"""
    mfi = data.MFI_14
    
    return mfi[0] > threshold and mfi[-1] <= threshold

def signal_mfi_bullish(data, lower=40, upper=80):
    """MFI處於多頭區間"""
    return lower < data.MFI_14[0] < upper

def signal_roc_positive(data):
    """變動率指標轉正"""
    return data.ROC_12[0] > 0 and data.ROC_12[-1] <= 0

def signal_stochrsi_golden_cross(data):
    """隨機RSI黃金交叉"""
    k = data.STOCHRSI_K
    d = data.STOCHRSI_D
    
    return k[0] > d[0] and k[-1] <= d[-1]

def signal_stochrsi_oversold_cross(data, threshold=0.2):
    """隨機RSI超賣後向上突破"""
    k = data.STOCHRSI_K
    
    return k[0] > threshold and k[-1] <= threshold

# ------------------- 出場策略 ------------------ #

def exit_ma_death_cross(data, fast_period=5, slow_period=20):
    """均線死亡交叉：快線向下跌破慢線"""
    fast_ma = f'SMA_{fast_period}' if f'SMA_{fast_period}' in data.lines.getlinealiases() else f'EMA_{fast_period}'
    slow_ma = f'SMA_{slow_period}' if f'SMA_{slow_period}' in data.lines.getlinealiases() else f'EMA_{slow_period}'
    
    fast_line = getattr(data, fast_ma)
    slow_line = getattr(data, slow_ma)
    
    # 當前快線 < 慢線 且 前一根快線 >= 慢線
    return fast_line[0] < slow_line[0] and fast_line[-1] >= slow_line[-1]

def exit_ma_bearish(data, fast_period=5, slow_period=20):
    """均線空頭排列：快線在慢線之下"""
    fast_ma = f'SMA_{fast_period}' if f'SMA_{fast_period}' in data.lines.getlinealiases() else f'EMA_{fast_period}'
    slow_ma = f'SMA_{slow_period}' if f'SMA_{slow_period}' in data.lines.getlinealiases() else f'EMA_{slow_period}'
    
    fast_line = getattr(data, fast_ma)
    slow_line = getattr(data, slow_ma)
    
    return fast_line[0] < slow_line[0]

def exit_price_below_ma(data, ma_period=20, ma_type='SMA'):
    """價格跌破均線"""
    ma_name = f'{ma_type}_{ma_period}'
    ma_line = getattr(data, ma_name)
    
    return data.close[0] < ma_line[0] and data.close[-1] >= ma_line[-1]

def exit_ma_slope_down(data, ma_period=20, ma_type='SMA', lookback=3):
    """均線向下傾斜"""
    ma_name = f'{ma_type}_{ma_period}'
    ma_line = getattr(data, ma_name)
    
    # 檢查最近N根K線均線持續下降
    return all(ma_line[-i] > ma_line[-i+1] for i in range(lookback, 0, -1))

def exit_rsi_overbought_cross(data, rsi_period=14, threshold=70):
    """RSI超買後向下跌破"""
    rsi_name = f'RSI_{rsi_period}'
    rsi = getattr(data, rsi_name)
    
    # RSI從高於threshold向下跌破
    return rsi[0] < threshold and rsi[-1] >= threshold

def exit_rsi_bearish(data, rsi_period=14, lower=30, upper=60):
    """RSI處於空頭區間"""
    rsi_name = f'RSI_{rsi_period}'
    rsi = getattr(data, rsi_name)
    
    return lower < rsi[0] < upper

def exit_macd_death_cross(data):
    """MACD死亡交叉：MACD線向下跌破信號線"""
    macd = data.MACD
    signal = data.MACD_signal
    
    return macd[0] < signal[0] and macd[-1] >= signal[-1]

def exit_macd_bearish(data):
    """MACD空頭：MACD線在信號線之下"""
    return data.MACD[0] < data.MACD_signal[0]

def exit_macd_hist_negative(data):
    """MACD柱狀圖轉負"""
    return data.MACD_hist[0] < 0 and data.MACD_hist[-1] >= 0

def exit_kd_death_cross(data):
    """KD死亡交叉：K線向下跌破D線"""
    k = data.KD_K
    d = data.KD_D
    
    return k[0] < d[0] and k[-1] >= d[-1]

def exit_kd_overbought_cross(data, threshold=80):
    """KD超買區死亡交叉"""
    k = data.KD_K
    d = data.KD_D
    
    # K線向下跌破D線且兩者都在超買區
    return (k[0] < d[0] and k[-1] >= d[-1] and 
            k[0] > threshold and d[0] > threshold)

def exit_kd_bearish(data):
    """KD空頭：K線在D線之下"""
    return data.KD_K[0] < data.KD_D[0]

def exit_bb_upper_bounce(data):
    """布林通道上軌反彈"""
    close = data.close
    bb_upper = data.BB_upper
    
    # 價格從上軌反彈
    return close[-1] >= bb_upper[-1] and close[0] < bb_upper[0]

def exit_bb_squeeze_break_down(data):
    """布林通道收縮後向下突破"""
    bandwidth = data.BB_bandwidth
    close = data.close
    bb_middle = data.BB_middle
    
    # 通道寬度收縮且價格跌破中軌
    return (bandwidth[0] < bandwidth[-5] and 
            close[0] < bb_middle[0] and close[-1] >= bb_middle[-1])

def exit_bb_percent_b_bearish(data, threshold=0.5):
    """布林%B指標空頭"""
    return data.BB_percent_b[0] < threshold

def exit_cci_overbought_cross(data, threshold=100):
    """CCI超買後向下跌破"""
    cci = data.CCI_20
    
    return cci[0] < threshold and cci[-1] >= threshold

def exit_cci_bearish(data):
    """CCI空頭區間"""
    return data.CCI_20[0] < 0

def exit_willr_overbought_cross(data, threshold=-20):
    """威廉指標超買後向下跌破"""
    willr = data.WILLR_14
    
    return willr[0] < threshold and willr[-1] >= threshold

def exit_mfi_overbought_cross(data, threshold=80):
    """資金流量指標超買後向下跌破"""
    mfi = data.MFI_14
    
    return mfi[0] < threshold and mfi[-1] >= threshold

def exit_mfi_bearish(data, lower=20, upper=60):
    """MFI處於空頭區間"""
    return lower < data.MFI_14[0] < upper

def exit_roc_negative(data):
    """變動率指標轉負"""
    return data.ROC_12[0] < 0 and data.ROC_12[-1] >= 0

def exit_stochrsi_death_cross(data):
    """隨機RSI死亡交叉"""
    k = data.STOCHRSI_K
    d = data.STOCHRSI_D
    
    return k[0] < d[0] and k[-1] >= d[-1]

def exit_stochrsi_overbought_cross(data, threshold=0.8):
    """隨機RSI超買後向下跌破"""
    k = data.STOCHRSI_K
    
    return k[0] < threshold and k[-1] >= threshold
