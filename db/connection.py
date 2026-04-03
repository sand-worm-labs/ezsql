import os
import psycopg2
from psycopg2.extras import RealDictCursor



def get_connection():
    DATABASE_URL = os.getenv("DATABASE_URL")
    return psycopg2.connect(DATABASE_URL)
