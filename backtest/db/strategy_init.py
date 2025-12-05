from sqlalchemy.orm import sessionmaker

from postgreSQL_session import db_session
from models import Strategy, BacktestHistory

# get postgreSQL session
print("Connecting to PostgreSQL...")
sess_db = db_session()
db = next(sess_db)

# 新增基本策略
new_strategy = Strategy(
    strategy_name='Baseline_metric',
    description='基本策略 - 技術面指標',
    strategy_func='MultiIndicatorStrategy',
    param_path='None'
)

db.add(new_strategy)
db.flush()              # id
db.refresh(new_strategy)    # created_at
db.commit()