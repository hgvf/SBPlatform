-- 啟用 pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 保存策略與 function name
CREATE TABLE IF NOT EXISTS tbl_strategy (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    strategy_func VARCHAR(100) NOT NULL,
    param_path TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 保存策略回測結果
CREATE TABLE IF NOT EXISTS tbl_backtest_history (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES tbl_strategy(id) ON DELETE CASCADE,
    backtest_name VARCHAR(50) NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    test_date DATE NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    market VARCHAR(5) NOT NULL,
    total_return REAL,
    max_drawdown REAL,
    sharpe_ratio REAL,
    trades_count INTEGER,
    report_path TEXT,
    custom_params_path TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
