import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base

def init_db():
    # environment variables
    load_dotenv("../.env", override=True)
    
    username = os.getenv("POSTGRES_USER", "")
    passwd = os.getenv("POSTGRES_PASSWORD", "")
    db_name = os.getenv("POSTGRES_DB", "")
    db_name = "postgres"
    db_port = os.getenv("POSTGRES_PORT", "5432")
    
    DATABASE_URL = f"postgresql+psycopg2://{username}:{passwd}@postgres:{db_port}/{db_name}"
    
    engine = create_engine(DATABASE_URL, echo=True)

    Base.metadata.create_all(bind=engine)

def db_session():
    # environment variables
    load_dotenv("../.env", override=True)
    
    username = os.getenv("POSTGRES_USER", "")
    passwd = os.getenv("POSTGRES_PASSWORD", "")
    db_name = os.getenv("POSTGRES_DB", "")
    db_name = "postgres"
    db_port = os.getenv("POSTGRES_PORT", "5432")
    
    DATABASE_URL = f"postgresql+psycopg2://{username}:{passwd}@postgres:{db_port}/{db_name}"

    engine = create_engine(DATABASE_URL, echo=True)
    SessionLocal = sessionmaker(bind=engine)

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
