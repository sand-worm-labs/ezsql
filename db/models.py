from math import log
from typing import Optional
import uuid
from pydantic import BaseModel, Field
from datetime import datetime

class ToolInput(BaseModel):
    key:      str
    label:    str
    type:     str   # address|token|protocol|chain|date_range|number|text
    required: bool
    default:  Optional[str | int | float] = None
 
 
class ToolNode(BaseModel):
    tool_id:        str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    g1:             str
    g2:             str
    g3:             str
    g4:             str
    g5:             str
    description:    str
    inputs:         list[ToolInput]        = []
    source_queries: list[str]             = []
    usage_count:    int                   = 0
    last_used_at:   Optional[datetime]    = None
    embedding:      Optional[list[float]] = None
    created_at:     Optional[datetime]    = None
    updated_at:     Optional[datetime]    = None
 
    def to_llm_node(self) -> dict:
        """Slim node injected into system prompt at runtime. No taxonomy, no embedding."""
        return {
            "tool_id":     self.tool_id,
            "name":        self.g3,
            "description": self.description,
            "inputs":      [i.model_dump() for i in self.inputs],
        }
 
    def popularity_score(self) -> float:
        """Combined score for router re-ranking. Source count = prior, usage = signal."""
        source = log(len(self.source_queries) + 1)
        usage  = log(self.usage_count + 1)
        return source * 0.4 + usage * 0.6
 

class Domain(BaseModel):
    domain_id:   str
    name:        str
    description: str
 