from typing import Optional
from sqlmodel import Field, SQLModel

class Query(SQLModel, table=True):
    __tablename__: str = "QUERIES"

    query_id:     Optional[str] = Field(default=None, primary_key=True)
    name:         Optional[str] = None
    description:  Optional[str] = None
    tags:         Optional[str] = None
    version:      Optional[str] = None
    parameters:   Optional[str] = None
    query_engine: Optional[str] = None
    query_sql:    Optional[str] = None
    is_private:   Optional[str] = None
    is_archived:  Optional[str] = None
    is_unsaved:   Optional[str] = None
    owner:        Optional[str] = None