import backtrader as bt

class MySignalStrategy(bt.Strategy):
    """
    方法A：原生下單 buy()/close()。
    - 可用 SMA 交叉 或 外部 signal 欄位（1=買, -1=賣/平, 0/NaN=無）
    - 支援固定% 或 ATR*mult 停損、時間停損、期末強平
    - 記錄交易與簡單統計（配合 analyzers 產生完整報表）
    """
    params = dict(
        use_external_signal=False,   # True 時使用 data.signal 欄位
        fast=10,
        slow=20,
        atr_period=14,
        stop_loss_pct=None,          # 例如 0.05 → 5% 停損
        atr_stop_mult=None,          # 例如 2.0 → 2*ATR 停損
        time_stop_bars=None,         # N 根 bar 後平倉
        breakeven_arm_R=None,        # 1.0 表示 ≥1R 後啟動保本（此範例僅示警位）
        force_close_last=True,
    )

    def __init__(self):
        # 指標
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)
        if not self.p.use_external_signal:
            self.fast = bt.ind.SMA(self.data.close, period=self.p.fast)
            self.slow = bt.ind.SMA(self.data.close, period=self.p.slow)
            self.cross = bt.ind.CrossOver(self.fast, self.slow)

        # 交易狀態
        self.order = None
        self.entry_price = None
        self.entry_dt = None
        self.entry_size = 0
        self.bars_held = 0
        self.highest = None
        self.lowest = None
        self.risk_R = None
        self.trades = []

    def _signal(self):
        if self.p.use_external_signal:
            try:
                val = self.data.signal[0]
                if val != val:  # NaN
                    return 'hold'
                sig = int(val)
                if not self.position and sig == 1:
                    return 'buy'
                if self.position and sig == -1:
                    return 'sell'
                return 'hold'
            except Exception:
                return 'hold'
        else:
            if not self.position and self.cross > 0:
                return 'buy'
            if self.position and self.cross < 0:
                return 'sell'
            return 'hold'

    def next(self):
        if self.order:
            return

        sig = self._signal()

        if self.position:
            self.bars_held += 1
            c = float(self.data.close[0])
            self.highest = c if self.highest is None else max(self.highest, c)
            self.lowest = c if self.lowest is None else min(self.lowest, c)

            # 時間停損
            if self.p.time_stop_bars and self.bars_held >= self.p.time_stop_bars:
                self.order = self.close(exectype=bt.Order.Market)
                return

            # 出場訊號
            if sig == 'sell':
                self.order = self.close(exectype=bt.Order.Market)
                return

            # 固定% 或 ATR 停損
            sl_price = None
            if self.p.stop_loss_pct:
                sl_price = self.entry_price * (1 - self.p.stop_loss_pct)
            elif self.p.atr_stop_mult:
                sl_price = self.entry_price - float(self.atr[0]) * self.p.atr_stop_mult
            if sl_price is not None and c <= sl_price:
                self.order = self.close(exectype=bt.Order.Market)
                return
        else:
            if sig == 'buy':
                self.order = self.buy()
                return

    def notify_order(self, order):
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

                if self.p.stop_loss_pct:
                    self.risk_R = self.entry_price * self.p.stop_loss_pct
                elif self.p.atr_stop_mult:
                    self.risk_R = float(self.atr[0]) * self.p.atr_stop_mult
                else:
                    self.risk_R = None
            else:
                exit_price = float(order.executed.price)
                exit_dt = self.data.datetime.datetime(0)
                qty = abs(int(order.executed.size)) or self.entry_size

                gross_ret_pct = (exit_price / self.entry_price - 1) * 100.0 if self.entry_price else 0.0
                rr_multiple = ((exit_price - self.entry_price) / self.risk_R) if (self.entry_price and self.risk_R) else float('nan')

                self.trades.append(dict(
                    entry_time=self.entry_dt,
                    entry_price=self.entry_price,
                    exit_time=exit_dt,
                    exit_price=exit_price,
                    qty=qty,
                    gross_return_pct=gross_ret_pct,
                    holding_period=self.bars_held,
                    max_runup_pct=((self.highest / self.entry_price - 1) * 100.0) if self.highest and self.entry_price else 0.0,
                    max_drawdown_pct=((self.lowest / self.entry_price - 1) * 100.0) if self.lowest and self.entry_price else 0.0,
                    rr_multiple=rr_multiple,
                ))

                # reset
                self.entry_price = None
                self.entry_size = 0
                self.entry_dt = None
                self.bars_held = 0
                self.highest = None
                self.lowest = None
                self.risk_R = None

        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def notify_trade(self, trade):
        # 可在此把 trade.pnl, trade.pnlcomm 回填到 self.trades[-1]
        pass

    def stop(self):
        if self.p.force_close_last and self.position:
            self.close(exectype=bt.Order.Close)