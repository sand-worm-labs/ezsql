import os
import json
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import build_stage2_prompt
from .registry import get_domains, get_registry_json, upsert_tool


class GrimoireState(TypedDict):
    query_id:        str
    title:           str
    sql:             str
    domains:         List[dict]
    registry_json:   str
    classified_tool: Optional[dict]
    tool_id:         Optional[str]
    error:           Optional[str]


llm = ChatOpenRouter(
    model="moonshotai/kimi-k2",
    temperature=0,
    openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
)


def initialize_node(state: GrimoireState) -> GrimoireState:
    print("→ initialize")
    return {
        **state,
        "domains":       get_domains(),
        "registry_json": get_registry_json(),
        "error":         None,
    }


def classify_node(state: GrimoireState) -> GrimoireState:
    print("→ classify")
    prompt = build_stage2_prompt(
        title=state["title"],
        sql=state["sql"],
        registry_json=state["registry_json"],
        domains=state["domains"],
    )
    print(f"Prompt:\n{prompt}\n")
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        cleaned = response.content.replace("```json", "").replace("```", "").strip()
        tool_data = json.loads(cleaned)
        return {**state, "classified_tool": tool_data}
    except json.JSONDecodeError as e:
        return {**state, "error": f"classify parse error: {e}"}


def register_node(state: GrimoireState) -> GrimoireState:
    print("→ register")
    tool_id = upsert_tool(state["classified_tool"], source_query_id=state["query_id"])
    return {**state, "tool_id": tool_id}


def should_register(state: GrimoireState) -> str:
    return "end" if state.get("error") else "register"


workflow = StateGraph(GrimoireState)

workflow.add_node("initialize", initialize_node)
workflow.add_node("classify",   classify_node)
workflow.add_node("register",   register_node)

workflow.set_entry_point("initialize")
workflow.add_edge("initialize", "classify")
workflow.add_conditional_edges(
    "classify",
    should_register,
    {"register": "register", "end": END},
)
workflow.add_edge("register", END)

grimoire_pipeline = workflow.compile()


def run_grimoire_pipeline(query_id: str, title: str, sql: str) -> dict:
    return grimoire_pipeline.invoke({
        "query_id":        query_id,
        "title":           title,
        "sql":             sql,
        "domains":         [],
        "registry_json":   "",
        "classified_tool": None,
        "tool_id":         None,
        "error":           None,
    })