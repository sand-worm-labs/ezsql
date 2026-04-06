import uuid
import json
from math import log
from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel, Column
from pydantic import BaseModel

class ToolInput(BaseModel):
    key:      str
    label:    str
    type:     str
    required: bool
    default:  Optional[str | int | float] = None

class ToolNode(SQLModel, table=True):
    __tablename__ = "grimoire_tools"

    tool_id:      str            = Field(default_factory=lambda: str(uuid.uuid4())[:8], primary_key=True)
    g1:           str
    g2:           str
    g3:           str
    g4:           str
    g5:           str
    description:  str
    inputs:       str            = Field(default="[]")        # JSON string — ToolInput list
    source_queries: str          = Field(default="[]")        # JSON string — list[str]
    usage_count:  int            = Field(default=0)
    last_used_at: Optional[datetime] = None
    created_at:   Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at:   Optional[datetime] = Field(default_factory=datetime.utcnow)

    def get_inputs(self) -> list[ToolInput]:
        return [ToolInput(**i) for i in json.loads(self.inputs)]

    def set_inputs(self, inputs: list[ToolInput]):
        self.inputs = json.dumps([i.model_dump() for i in inputs])

    def get_source_queries(self) -> list[str]:
        return json.loads(self.source_queries)

    def add_source_query(self, query_id: str):
        current = self.get_source_queries()
        if query_id not in current:
            current.append(query_id)
            self.source_queries = json.dumps(current)

    def to_llm_node(self) -> dict:
        return {
            "tool_id":     self.tool_id,
            "name":        self.g3,
            "description": self.description,
            "inputs":      json.loads(self.inputs),
        }

    def popularity_score(self) -> float:
        source = log(len(self.get_source_queries()) + 1)
        usage  = log(self.usage_count + 1)
        return source * 0.4 + usage * 0.6


class Domain(SQLModel, table=True):
    __tablename__ = "grimoire_domains"

    domain_id:   str = Field(primary_key=True)
    name:        str = Field(unique=True)
    description: str
    created_at:  Optional[datetime] = Field(default_factory=datetime.utcnow)