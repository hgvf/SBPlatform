import backtrader as bt

class TWStockCommInfo(bt.CommInfoBase):
    """
    佣金/稅：
    - buy_fee_rate: 買進手續費（比例）
    - sell_fee_rate: 賣出手續費（比例）
    - sell_tax_rate: 賣出證交稅（比例）
    注意：percabs=True 代表按成交金額百分比
    """

    params = dict(
        buy_fee_rate=0.001425,
        sell_fee_rate=0.001425,
        sell_tax_rate=0.003,
        stocklike=True,
        percabs=True,
    )

    def getcommission(self, size, price):
        gross = abs(size) * price

        if size > 0: # buy
            return gross * self.p.buy_fee_rate
        else: # sell
            return gross * (self.p.sell_fee_rate + self.p.sell_tax_rate)

class ATRRiskSizer(bt.Sizer):
    """以 ATR*mult 停損距離推算單筆風險等權口數。需要策略內有 atr 指標。"""
    
    params = dict(risk_perc=0.02)

    def _getsizing(self, comminfo, cash, data, isbuy):
        strat = self.strategy
        atr_val = float(getattr(strat, 'atr', [None])[0] or 0)
        stop_mult = getattr(strat.p, 'atr_stop_mult', None)
        price = float(data.close[0]) if len(data) else 0
        
        if atr_val <= 0 or not stop_mult or price <= 0:
            return 0
        
        stop_dist = atr_val * stop_mult
        risk_budget = cash * self.p.risk_perc
        size = risk_budget / stop_dist

        return int(max(size, 0))