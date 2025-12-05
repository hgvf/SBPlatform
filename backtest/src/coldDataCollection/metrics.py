import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union


class TechnicalIndicatorCalculator:
    """技術指標計算器類，提供完整的技術分析指標計算功能"""
    
    def __init__(self):
        """初始化技術指標計算器"""
        pass
    
    # ---------------------------
    # 私有輔助方法
    # ---------------------------
    def _sma(self, s: pd.Series, window: int) -> pd.Series:
        """簡單移動平均線"""
        return s.rolling(window, min_periods=window).mean()

    def _ema(self, s: pd.Series, span: int) -> pd.Series:
        """指數移動平均線"""
        return s.ewm(span=span, adjust=False, min_periods=span).mean()

    def _wilder_rma(self, s: pd.Series, window: int) -> pd.Series:
        """Wilder's RMA (用於RSI計算)"""
        alpha = 1.0 / window
        return s.ewm(alpha=alpha, adjust=False, min_periods=window).mean()

    def _true_range(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """真實波動幅度"""
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr

    def _hlc3(self, df: pd.DataFrame) -> pd.Series:
        """典型價格 (High + Low + Close) / 3"""
        return (df['High'] + df['Low'] + df['Close']) / 3.0

    def _ohlc4(self, df: pd.DataFrame) -> pd.Series:
        """平均價格 (Open + High + Low + Close) / 4"""
        return (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4.0

    # ---------------------------
    # 移動平均線指標
    # ---------------------------
    def sma(self, df: pd.DataFrame, window: int, price_col: str = 'Close') -> pd.Series:
        """簡單移動平均線"""
        return self._sma(df[price_col], window).rename(f"SMA_{window}")

    def ema(self, df: pd.DataFrame, window: int, price_col: str = 'Close') -> pd.Series:
        """指數移動平均線"""
        return self._ema(df[price_col], window).rename(f"EMA_{window}")

    # ---------------------------
    # 動量指標
    # ---------------------------
    def rsi(self, df: pd.DataFrame, window: int = 14, price_col: str = 'Close') -> pd.Series:
        """相對強弱指標 (RSI)"""
        delta = df[price_col].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = self._wilder_rma(gain, window)
        avg_loss = self._wilder_rma(loss, window)
        rs = avg_gain / avg_loss.replace(0, np.nan)
        out = 100.0 - (100.0 / (1.0 + rs))
        return out.rename(f"RSI_{window}")

    def macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, price_col: str = 'Close') -> pd.DataFrame:
        """MACD指標 (回傳 macd/signal/hist)"""
        ema_fast = self._ema(df[price_col], fast)
        ema_slow = self._ema(df[price_col], slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, signal)
        hist = macd_line - signal_line
        return pd.DataFrame({
            'MACD_macd': macd_line,
            'MACD_signal': signal_line,
            'MACD_hist': hist
        })

    def stoch_kd(self, df: pd.DataFrame, k_period: int = 9, d_period: int = 3, smooth_k: int = 3) -> pd.DataFrame:
        """KD隨機指標 (Stochastic %K/%D)"""
        lowest_low = df['Low'].rolling(k_period, min_periods=k_period).min()
        highest_high = df['High'].rolling(k_period, min_periods=k_period).max()
        raw_k = 100.0 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
        k = raw_k.rolling(smooth_k, min_periods=smooth_k).mean()
        d = k.rolling(d_period, min_periods=d_period).mean()
        return pd.DataFrame({
            'KD_K': k.rename(f'K_{k_period}'), 
            'KD_D': d.rename(f'D_{d_period}')
        })

    def stoch_rsi(self, df: pd.DataFrame, rsi_window: int = 14, stoch_window: int = 14, 
                  k: int = 3, d: int = 3, price_col: str = 'Close') -> pd.DataFrame:
        """隨機RSI指標"""
        r = self.rsi(df, window=rsi_window, price_col=price_col)
        rmin = r.rolling(stoch_window, min_periods=stoch_window).min()
        rmax = r.rolling(stoch_window, min_periods=stoch_window).max()
        stoch = (r - rmin) / (rmax - rmin)
        kline = stoch.rolling(k, min_periods=k).mean() * 100.0
        dline = kline.rolling(d, min_periods=d).mean()
        return pd.DataFrame({
            'STOCHRSI_K': kline.rename(f'STOCHRSI_K'), 
            'STOCHRSI_D': dline.rename(f'STOCHRSI_D')
        })

    def roc(self, df: pd.DataFrame, window: int = 12, price_col: str = 'Close') -> pd.Series:
        """變動率指標 (Rate of Change)"""
        s = df[price_col]
        out = (s / s.shift(window) - 1.0) * 100.0
        return out.rename(f'ROC_{window}')

    # ---------------------------
    # 波動性指標
    # ---------------------------
    def bbands(self, df: pd.DataFrame, window: int = 20, stdev: float = 2.0, price_col: str = 'Close') -> pd.DataFrame:
        """布林通道 (Bollinger Bands)"""
        mid = self._sma(df[price_col], window)
        std = df[price_col].rolling(window, min_periods=window).std()
        upper = mid + stdev * std
        lower = mid - stdev * std
        # 也回傳 %B 與 Bandwidth，方便策略
        percent_b = (df[price_col] - lower) / (upper - lower)
        bandwidth = (upper - lower) / mid
        return pd.DataFrame({
            'BB_upper': upper.rename(f'BBU_{window}_{stdev}'),
            'BB_middle': mid.rename(f'BBM_{window}'),
            'BB_lower': lower.rename(f'BBL_{window}_{stdev}'),
            'BB_percent_b': percent_b.rename(f'%B_{window}_{stdev}'),
            'BB_bandwidth': bandwidth.rename(f'BBW_{window}_{stdev}')
        })

    def cci(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """商品通道指標 (Commodity Channel Index)"""
        tp = self._hlc3(df)
        sma_tp = self._sma(tp, window)
        mad = (tp - sma_tp).abs().rolling(window, min_periods=window).mean()
        cci_val = (tp - sma_tp) / (0.015 * mad)
        return cci_val.rename(f'CCI_{window}')

    def williams_r(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """威廉指標 (Williams %R)"""
        highest_high = df['High'].rolling(window, min_periods=window).max()
        lowest_low = df['Low'].rolling(window, min_periods=window).min()
        r = -100.0 * (highest_high - df['Close']) / (highest_high - lowest_low)
        return r.rename(f'WILLR_{window}')

    # ---------------------------
    # 成交量指標
    # ---------------------------
    def mfi(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """資金流量指標 (Money Flow Index)"""
        tp = self._hlc3(df)
        mf = tp * df['Volume']
        pos_mf = np.where(tp > tp.shift(1), mf, 0.0)
        neg_mf = np.where(tp < tp.shift(1), mf, 0.0)
        pos_sum = pd.Series(pos_mf, index=df.index).rolling(window, min_periods=window).sum()
        neg_sum = pd.Series(neg_mf, index=df.index).rolling(window, min_periods=window).sum()
        mr = pos_sum / neg_sum.replace(0, np.nan)
        out = 100.0 - (100.0 / (1.0 + mr))
        return out.rename(f'MFI_{window}')

    # ---------------------------
    # 批量計算方法
    # ---------------------------
    def compute_all_indicators(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        ma_windows: Tuple[int, ...] = (5, 20, 60, 120, 240),
        rsi_window: Tuple[int, ...] = (7, 14),
        macd_params: Tuple[int, int, int] = (12, 26, 9),
        stoch_params: Tuple[int, int, int] = (9, 3, 3),
        bb_params: Tuple[int, float] = (20, 2.0),
        include_indicators: Optional[List[str]] = None
    ) -> Dict[str, pd.Series]:
        """
        計算所有常用技術指標
        
        參數:
        - df: OHLCV數據框
        - price_col: 價格欄位名稱
        - ma_windows: 移動平均線週期
        - rsi_window: RSI週期
        - macd_params: MACD參數 (fast, slow, signal)
        - stoch_params: KD參數 (k_period, d_period, smooth_k)
        - bb_params: 布林通道參數 (window, stdev)
        - include_indicators: 指定要計算的指標列表，None表示計算全部
        
        回傳: Dict[str, pd.Series] 指標名稱對應的Series
        """
        indicators = {}
        
        # 如果沒有指定要計算的指標，則計算全部
        if include_indicators is None:
            include_indicators = ['ma', 'rsi', 'macd', 'stoch', 'bb', 'cci', 'willr', 'mfi', 'roc', 'stochrsi']
        
        # 移動平均線
        if 'ma' in include_indicators:
            for window in ma_windows:
                indicators[f"SMA_{window}"] = self.sma(df, window, price_col)
                indicators[f"EMA_{window}"] = self.ema(df, window, price_col)
        
        # RSI
        if 'rsi' in include_indicators:
            for window in rsi_window:
                indicators[f"RSI_{window}"] = self.rsi(df, window, price_col)
        
        # MACD
        if 'macd' in include_indicators:
            macd_df = self.macd(df, *macd_params, price_col=price_col)
            indicators.update(macd_df.to_dict('series'))
        
        # KD隨機指標
        if 'stoch' in include_indicators:
            kd_df = self.stoch_kd(df, *stoch_params)
            indicators.update(kd_df.to_dict('series'))
        
        # 布林通道
        if 'bb' in include_indicators:
            bb_df = self.bbands(df, *bb_params, price_col=price_col)
            indicators.update(bb_df.to_dict('series'))
        
        # CCI
        if 'cci' in include_indicators:
            indicators["CCI_20"] = self.cci(df, 20)
        
        # 威廉指標
        if 'willr' in include_indicators:
            indicators["WILLR_14"] = self.williams_r(df, 14)
        
        # 資金流量指標
        if 'mfi' in include_indicators:
            indicators["MFI_14"] = self.mfi(df, 14)
        
        # 變動率指標
        if 'roc' in include_indicators:
            indicators["ROC_12"] = self.roc(df, 12, price_col)
        
        # 隨機RSI
        if 'stochrsi' in include_indicators:
            stochrsi_df = self.stoch_rsi(df, 14, 14, 3, 3, price_col)
            indicators.update(stochrsi_df.to_dict('series'))
        
        return indicators

    def add_indicators_to_dataframe(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        ma_windows: Tuple[int, ...] = (5, 20, 60, 120, 240),
        rsi_window: Tuple[int, ...] = (7, 14),
        macd_params: Tuple[int, int, int] = (12, 26, 9),
        stoch_params: Tuple[int, int, int] = (9, 3, 3),
        bb_params: Tuple[int, float] = (20, 2.0),
        include_indicators: Optional[List[str]] = None,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        將技術指標添加到原始數據框中
        
        參數:
        - df: 原始OHLCV數據框
        - 其他參數同 compute_all_indicators
        - inplace: 是否在原數據框上直接修改
        
        回傳: 包含技術指標的數據框
        """
        # 計算所有指標
        indicators = self.compute_all_indicators(
            df, price_col, ma_windows, rsi_window, macd_params, 
            stoch_params, bb_params, include_indicators
        )
        
        # 決定是否在原數據框上操作
        if inplace:
            result_df = df
        else:
            result_df = df.copy()
        
        # 添加指標到數據框
        for name, series in indicators.items():
            result_df[name] = series
        
        return result_df

    def to_metric_long_format(
        self,
        indicators: Dict[str, pd.Series],
        ts_index: pd.Index,
        security_id: int,
        interval: str,
        name_component_map: Optional[Dict[str, Tuple[str, str]]] = None
    ) -> pd.DataFrame:
        """
        轉換指標為長格式 (ts/component/value)
        
        參數:
        - indicators: 指標字典
        - ts_index: 時間索引
        - security_id: 證券ID
        - interval: 時間間隔
        - name_component_map: 名稱組件映射
        
        回傳: 長格式的數據框
        """
        records = []
        
        for col, series in indicators.items():
            # 解析名稱和組件
            if name_component_map and col in name_component_map:
                name, component = name_component_map[col]
            else:
                if '_' in col and any(col.startswith(p) for p in ['MACD', 'BB', 'KD', 'STOCHRSI']):
                    name, component = col.split('_', 1)
                else:
                    name, component = (col.split('_', 1)[0], 'value')
            
            # 添加數據記錄
            for ts, value in series.items():
                if pd.isna(value):
                    continue
                records.append({
                    'security_id': security_id,
                    'interval': interval,
                    'ts': pd.Timestamp(ts).to_pydatetime(),
                    'name': name,
                    'component': component,
                    'value': float(value)
                })
        
        return pd.DataFrame.from_records(
            records, 
            columns=['security_id', 'interval', 'ts', 'name', 'component', 'value']
        )

    def get_available_indicators(self) -> Dict[str, List[str]]:
        """
        獲取可用的技術指標列表
        
        回傳: 按類別分組的指標列表
        """
        return {
            '移動平均線': ['SMA', 'EMA'],
            '動量指標': ['RSI', 'MACD', 'KD', 'StochRSI', 'ROC'],
            '波動性指標': ['Bollinger Bands', 'CCI', 'Williams %R'],
            '成交量指標': ['MFI']
        }
