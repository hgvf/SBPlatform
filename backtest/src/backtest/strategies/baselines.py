import backtrader as bt
import sys
import inspect

from .metric_baseline_funcs import *

def _build_func_map():
    """
    自動建立 str -> function 對照表
    所有以 signal_ 或 exit_ 開頭的函式都會被收錄。
    """

    current_module = sys.modules[__name__]
    func_map = {}
    for name, func in inspect.getmembers(current_module, inspect.isfunction):
        if name.startswith("signal_") or name.startswith("exit_"):
            func_map[name] = func
    return func_map

class MultiIndicatorStrategy(bt.Strategy):
    """
    通用多指標組合策略（整合停損停利功能）
    
    使用方法：
    cerebro.addstrategy(
        MultiIndicatorStrategy,
        entry_signals=[signal_ma_golden_cross, signal_kd_golden_cross],
        entry_params=[{'fast_period': 5, 'slow_period': 10}, {}],
        exit_signals=[signal_ma_death_cross],
        exit_params=[{'fast_period': 5, 'slow_period': 10}],
        require_all_entry=True,
        require_any_exit=True,
        stop_loss_pct=0.05,
        atr_stop_mult=2.0,
        take_profit_pct=0.10,
        time_stop_bars=20,
    )
    """

    

    params = (
        # 進場信號
        ('entry_signals', []),
        ('entry_params', []),
        ('require_all_entry', True),  # True=所有進場信號都要滿足
        
        # 出場信號
        ('exit_signals', []),
        ('exit_params', []),
        ('require_any_exit', True),   # True=任一出場信號滿足即出場
        
        # 停損設定
        ('stop_loss_pct', None),      # 固定百分比停損 (例如 0.05 = 5%)
        ('atr_period', 14),
        ('atr_stop_mult', None),      # ATR倍數停損 (例如 2.0 = 2倍ATR)
        
        # 停利設定
        ('take_profit_pct', None),    # 固定百分比停利 (例如 0.10 = 10%)
        ('trailing_stop_pct', None),  # 移動停損 (例如 0.03 = 從最高點回撤3%)
        
        # 時間停損
        ('time_stop_bars', None),     # N根K線後強制出場
        
        # 其他設定
        ('breakeven_arm_R', None),    # 達到多少R後移動停損到保本 (例如 1.0)
        ('force_close_last', True),   # 回測結束時強制平倉
        ('printlog', True),
    )
    
    def __init__(self):
        self.FUNC_MAP = _build_func_map()

        # ATR指標（用於ATR停損）
        if self.params.atr_stop_mult:
            self.atr = bt.ind.ATR(self.data, period=self.params.atr_period)
        else:
            self.atr = None
        
        # 驗證參數
        if not self.params.entry_params:
            self.params.entry_params = [{}] * len(self.params.entry_signals)
        if not self.params.exit_params:
            self.params.exit_params = [{}] * len(self.params.exit_signals)
        
        if len(self.params.entry_signals) != len(self.params.entry_params):
            raise ValueError("entry_signals 和 entry_params 長度必須一致")
        if len(self.params.exit_signals) != len(self.params.exit_params):
            raise ValueError("exit_signals 和 exit_params 長度必須一致")
        
        # 交易狀態
        self.order = None
        self.entry_price = None
        self.entry_dt = None
        self.entry_size = 0
        self.bars_held = 0
        self.highest = None
        self.lowest = None
        self.risk_R = None
        self.stop_loss_price = None
        self.breakeven_activated = False
        
        # 交易記錄
        self.trades = []
    
    def log(self, txt, dt=None):
        """日誌輸出"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')
    
    def check_entry_signals(self):
        """檢查所有進場信號"""
        if not self.params.entry_signals:
            return False
        
        results = []
        for signal_func, params in zip(self.params.entry_signals, self.params.entry_params):
            try:
                result = self.FUNC_MAP[signal_func](self.datas[0], **params)
                results.append(result)
            except Exception as e:
                self.log(f'進場信號計算錯誤 {self.FUNC_MAP[signal_func].__name__}: {str(e)}')
                results.append(False)
        
        if self.params.require_all_entry:
            return all(results)
        else:
            return any(results)
    
    def check_exit_signals(self):
        """檢查所有出場信號"""
        if not self.params.exit_signals:
            return False
        
        results = []
        for signal_func, params in zip(self.params.exit_signals, self.params.exit_params):
            try:
                result = self.FUNC_MAP[signal_func](self.datas[0], **params)
                results.append(result)
            except Exception as e:
                self.log(f'出場信號計算錯誤 {self.FUNC_MAP[signal_func].__name__}: {str(e)}')
                results.append(False)
        
        if self.params.require_any_exit:
            return any(results)
        else:
            return all(results)
    
    def calculate_stop_loss(self):
        """計算停損價格"""
        if self.params.stop_loss_pct:
            return self.entry_price * (1 - self.params.stop_loss_pct)
        elif self.params.atr_stop_mult and self.atr:
            return self.entry_price - float(self.atr[0]) * self.params.atr_stop_mult
        return None
    
    def update_trailing_stop(self, current_price):
        """更新移動停損"""
        if not self.params.trailing_stop_pct:
            return
        
        if self.highest is None:
            self.highest = current_price
        else:
            self.highest = max(self.highest, current_price)
        
        # 計算移動停損價格
        trailing_stop = self.highest * (1 - self.params.trailing_stop_pct)
        
        # 更新停損價格（只能往上，不能往下）
        if self.stop_loss_price is None or trailing_stop > self.stop_loss_price:
            self.stop_loss_price = trailing_stop
    
    def check_breakeven(self, current_price):
        """檢查是否達到保本條件"""
        if (self.params.breakeven_arm_R and 
            not self.breakeven_activated and 
            self.risk_R):
            
            profit = current_price - self.entry_price
            r_multiple = profit / self.risk_R
            
            if r_multiple >= self.params.breakeven_arm_R:
                self.stop_loss_price = self.entry_price
                self.breakeven_activated = True
                self.log(f'保本停損啟動 - Stop: {self.stop_loss_price:.2f}')
    
    def next(self):
        """主邏輯"""
        if self.order:
            return
        
        current_price = float(self.data.close[0])
        
        # 持倉管理
        if self.position:
            self.bars_held += 1
            
            # 更新最高/最低價
            if self.highest is None:
                self.highest = current_price
            else:
                self.highest = max(self.highest, current_price)
            
            if self.lowest is None:
                self.lowest = current_price
            else:
                self.lowest = min(self.lowest, current_price)
            
            # 更新移動停損
            self.update_trailing_stop(current_price)
            
            # 檢查保本條件
            self.check_breakeven(current_price)
            
            # 1. 檢查停損
            if self.stop_loss_price and current_price <= self.stop_loss_price:
                self.log(f'停損出場 - Price: {current_price:.2f}, Stop: {self.stop_loss_price:.2f}')
                self.order = self.close(exectype=bt.Order.Market)
                return
            
            # 2. 檢查停利
            if self.params.take_profit_pct:
                take_profit_price = self.entry_price * (1 + self.params.take_profit_pct)
                if current_price >= take_profit_price:
                    self.log(f'停利出場 - Price: {current_price:.2f}, Target: {take_profit_price:.2f}')
                    self.order = self.close(exectype=bt.Order.Market)
                    return
            
            # 3. 檢查時間停損
            if self.params.time_stop_bars and self.bars_held >= self.params.time_stop_bars:
                self.log(f'時間停損出場 - 持有 {self.bars_held} 根K線')
                self.order = self.close(exectype=bt.Order.Market)
                return
            
            # 4. 檢查技術指標出場信號
            if self.check_exit_signals():
                self.log(f'技術指標出場信號 - Price: {current_price:.2f}')
                self.order = self.close(exectype=bt.Order.Market)
                return
        
        # 進場邏輯
        else:
            if self.check_entry_signals():
                self.log(f'進場信號觸發 - Price: {current_price:.2f}')
                self.order = self.buy()
    
    def notify_order(self, order):
        """訂單通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status == order.Completed:
            if order.isbuy():
                self.entry_price = float(order.executed.price)
                self.entry_size = int(order.executed.size)
                self.entry_dt = self.data.datetime.datetime(0)
                self.bars_held = 0
                self.highest = self.entry_price
                self.lowest = self.entry_price
                self.breakeven_activated = False
                
                # 計算初始停損
                self.stop_loss_price = self.calculate_stop_loss()
                
                # 計算R值
                if self.stop_loss_price:
                    self.risk_R = self.entry_price - self.stop_loss_price
                else:
                    self.risk_R = None
                
                stop_str = f"{self.stop_loss_price:.2f}" if self.stop_loss_price else "None"
                self.log(f'買入成交 - Price: {order.executed.price:.2f}, '
                        f'Size: {order.executed.size}, '
                        f'Stop: {stop_str}')
            
            else:  # 賣出
                exit_price = float(order.executed.price)
                exit_dt = self.data.datetime.datetime(0)
                qty = abs(int(order.executed.size)) or self.entry_size
                
                # 計算績效指標
                gross_ret_pct = ((exit_price / self.entry_price - 1) * 100.0 
                                if self.entry_price else 0.0)
                
                rr_multiple = ((exit_price - self.entry_price) / self.risk_R 
                              if (self.entry_price and self.risk_R) else float('nan'))
                
                max_runup_pct = ((self.highest / self.entry_price - 1) * 100.0 
                                if self.highest and self.entry_price else 0.0)
                
                max_drawdown_pct = ((self.lowest / self.entry_price - 1) * 100.0 
                                   if self.lowest and self.entry_price else 0.0)
                
                # 記錄交易
                self.trades.append(dict(
                    entry_time=self.entry_dt,
                    entry_price=self.entry_price,
                    exit_time=exit_dt,
                    exit_price=exit_price,
                    qty=qty,
                    gross_return_pct=gross_ret_pct,
                    holding_period=self.bars_held,
                    max_runup_pct=max_runup_pct,
                    max_drawdown_pct=max_drawdown_pct,
                    rr_multiple=rr_multiple,
                ))
                
                self.log(f'賣出成交 - Price: {exit_price:.2f}, '
                        f'Return: {gross_ret_pct:.2f}%, '
                        f'R: {rr_multiple:.2f}')
                
                # 重置狀態
                self.entry_price = None
                self.entry_size = 0
                self.entry_dt = None
                self.bars_held = 0
                self.highest = None
                self.lowest = None
                self.risk_R = None
                self.stop_loss_price = None
                self.breakeven_activated = False
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('訂單取消/保證金不足/拒絕')
        
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None
    
    def notify_trade(self, trade):
        """交易通知"""
        if not trade.isclosed:
            return
        
        if self.trades:
            self.trades[-1]['pnl'] = trade.pnl
            self.trades[-1]['pnl_comm'] = trade.pnlcomm
    
    def stop(self):
        """回測結束"""
        if self.params.force_close_last and self.position:
            self.log('期末強制平倉')
            self.close(exectype=bt.Order.Close)
        
        # 輸出交易統計
        if self.trades:
            total_trades = len(self.trades)
            winning_trades = sum(1 for t in self.trades if t['gross_return_pct'] > 0)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            avg_return = sum(t['gross_return_pct'] for t in self.trades) / total_trades
            avg_holding = sum(t['holding_period'] for t in self.trades) / total_trades
            
            self.log(f'\n=== 交易統計 ===')
            self.log(f'總交易次數: {total_trades}')
            self.log(f'勝率: {win_rate:.2f}%')
            self.log(f'平均報酬率: {avg_return:.2f}%')
            self.log(f'平均持有期間: {avg_holding:.1f} 根K線')

"""
# 範例1: 均線黃金交叉進場 + 死亡交叉出場 + 5%停損 + 10%停利
cerebro.addstrategy(
    MultiIndicatorStrategy,
    entry_signals=[signal_ma_golden_cross],
    entry_params=[{'fast_period': 5, 'slow_period': 20}],
    exit_signals=[signal_ma_death_cross],
    exit_params=[{'fast_period': 5, 'slow_period': 20}],
    stop_loss_pct=0.05,
    take_profit_pct=0.10,
    printlog=True
)

# 範例2: 多重進場條件 + ATR停損 + 移動停損
cerebro.addstrategy(
    MultiIndicatorStrategy,
    entry_signals=[
        signal_ma_golden_cross,
        signal_kd_oversold_cross,
        signal_rsi_bullish
    ],
    entry_params=[
        {'fast_period': 5, 'slow_period': 20},
        {'threshold': 20},
        {'rsi_period': 14, 'lower': 40, 'upper': 70}
    ],
    exit_signals=[
        signal_kd_death_cross,
        signal_rsi_overbought
    ],
    exit_params=[{}, {'rsi_period': 14, 'threshold': 70}],
    require_all_entry=True,     # 所有進場信號都要滿足
    require_any_exit=True,      # 任一出場信號即出場
    atr_stop_mult=2.0,          # 2倍ATR停損
    trailing_stop_pct=0.03,     # 3%移動停損
    time_stop_bars=20,          # 20根K線時間停損
    breakeven_arm_R=1.5,        # 達到1.5R後移動到保本
)

# 範例3: 簡單策略 - MACD + KD雙黃金交叉
cerebro.addstrategy(
    MultiIndicatorStrategy,
    entry_signals=[
        signal_macd_golden_cross,
        signal_kd_golden_cross
    ],
    entry_params=[{}, {}],
    exit_signals=[
        signal_macd_death_cross
    ],
    exit_params=[{}],
    stop_loss_pct=0.03,
    take_profit_pct=0.08,
)

# 範例4: 布林通道策略
cerebro.addstrategy(
    MultiIndicatorStrategy,
    entry_signals=[signal_bb_lower_bounce],
    entry_params=[{}],
    exit_signals=[signal_bb_upper_touch],
    exit_params=[{}],
    stop_loss_pct=0.05,
    trailing_stop_pct=0.02,
)
"""