from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, TIMESTAMP, ForeignKey
from sqlalchemy.orm import sessionmaker
import os

DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/trafficking")

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)
metadata = MetaData()

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True),
    Column("email", String, unique=True, nullable=False),
    Column("password_hash", String, nullable=False),
    Column("role", String, nullable=False),
    Column("created_at", TIMESTAMP)
)

uploads = Table(
    "uploads", metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("filename", String),
    Column("upload_time", TIMESTAMP)
)

def init_db():
    metadata.create_all(engine)

if __name__ == "__main__":
    init_db()
    print("Database initialized.")