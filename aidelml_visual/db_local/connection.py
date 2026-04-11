import os
from sqlmodel import create_engine, Session
from contextlib import contextmanager

engine = create_engine(os.getenv("DATABASE_URL"), pool_size=5, max_overflow=10)

@contextmanager
def get_session():
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise