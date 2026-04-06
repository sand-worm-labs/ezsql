from typing import TypedDict, List
import json
from langgraph.graph import StateGraph, END
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import build_stage1_prompt, build_stage2_prompt
from .registry import get_domains, get_registry_json, find_by_g3, upsert_tool
# from .embedder import embed_tool_description # Import when ready


class GrimoireState(TypedDict):
    # Input
    user_id: str
    raw_title: str
    raw_sql: str
    # Discovery context (loaded once)
    domains: List[dict]        # [{name, description}, ...]
    registry_context: str      # JSON string of existing tools
    # Pipeline artifacts
    current_units: List[dict]   # Stage 1 output
    processed_tools: List[dict] # Stage 2 output
    # Control
    errors: List[str]
    retry_count: int


llm = ChatOpenRouter(model="gpt-4-turbo", temperature=0)


def initialize_context_node(state: GrimoireState) -> GrimoireState:
    """Load taxonomy and registry context from Postgres."""
    print("--- [Node: Initialize Context] ---")
    domains = get_domains()            
    registry_json = get_registry_json()

    return {
        **state,
        "domains": domains,
        "registry_context": registry_json,
        "errors": [],
        "retry_count": 0,
        "processed_tools": [],
    }


def stage1_decompose_node(state: GrimoireState) -> GrimoireState:
    """Decompose raw SQL into logical units."""
    print("--- [Node: Stage 1 - Decompose] ---")

    prompt_text = build_stage1_prompt(state['raw_title'], state['raw_sql'])
    messages = [SystemMessage(content="You are an expert SQL decompiler."), HumanMessage(content=prompt_text)]

    response = llm.invoke(messages)

    try:
        cleaned = response.content.replace("```json", "").replace("```", "").strip()
        units = json.loads(cleaned).get("units", [])
        return {**state, "current_units": units}
    except json.JSONDecodeError:
        error_msg = f"Stage 1 failed to generate valid JSON: {response.content[:100]}..."
        return {**state, "errors": [error_msg]}


def stage2_classify_node(state: GrimoireState) -> GrimoireState:
    """Classify each unit into the taxonomy and extract inputs."""
    print(f"--- [Node: Stage 2 - Classify {len(state['current_units'])} units] ---")

    classified_tools = []
    errors = []

    for unit in state['current_units']:
        prompt_text = build_stage2_prompt(
            label=unit['label'],
            sql=unit['sql'],
            registry_json=state['registry_context'],
            domains=state['domains'],
        )

        messages = [SystemMessage(content="You are an expert onchain taxonomy engineer."), HumanMessage(content=prompt_text)]
        response = llm.invoke(messages)

        try:
            cleaned = response.content.replace("```json", "").replace("```", "").strip()
            tool_data = json.loads(cleaned)
            tool_data['sql_code'] = unit['sql']
            classified_tools.append(tool_data)
        except json.JSONDecodeError:
            errors.append(f"Stage 2 failed for unit '{unit['label']}': {response.content[:100]}...")

    return {**state, "processed_tools": classified_tools, "errors": errors}


def registry_match_node(state: GrimoireState) -> GrimoireState:
    """Check processed tools against Postgres for idempotency."""
    print("--- [Node: Registry Match] ---")

    for tool in state['processed_tools']:
        existing = find_by_g3(tool['g3'])
        if existing:
            tool['id'] = existing['tool_id']
            tool['note'] = "Matched existing tool"
        else:
            tool['id'] = None

    return state


def finalize_tool_node(state: GrimoireState) -> GrimoireState:
    """Upsert tools into Postgres (new insert or append source query)."""
    print("--- [Node: Finalize & Save] ---")

    source_query_id = state['raw_title']
    saved_ids = []

    for tool in state['processed_tools']:
        tool_id = upsert_tool(tool, source_query_id)

        # TODO: Generate vector embedding of g3+g4 and store to pgvector
        # description = f"{tool['g3']} - {tool['g4']}"
        # embed_tool_description(tool_id, description)

        saved_ids.append(tool_id)

    print(f"Pipeline complete. {len(saved_ids)} tools processed.")
    return state


# --- CONDITIONAL ROUTING -------------------------------------------------------

def should_continue(state: GrimoireState):
    if state['errors']:
        print(f"!!! Pipeline Error: {state['errors'][0]}")
        return "end"
    if state['retry_count'] > 3:
        print("!!! Max Retries Exceeded.")
        return "end"
    return "continue"


# --- GRAPH CONSTRUCTION --------------------------------------------------------

workflow = StateGraph(GrimoireState)

workflow.add_node("initialize", initialize_context_node)
workflow.add_node("stage1_decompose", stage1_decompose_node)
workflow.add_node("stage2_classify", stage2_classify_node)
workflow.add_node("registry_match", registry_match_node)
workflow.add_node("finalize", finalize_tool_node)

workflow.set_entry_point("initialize")

workflow.add_edge("initialize", "stage1_decompose")
workflow.add_edge("stage1_decompose", "stage2_classify")
workflow.add_edge("stage2_classify", "registry_match")
workflow.add_edge("registry_match", "finalize")
workflow.add_edge("finalize", END)

grimoire_pipeline = workflow.compile()


# --- EXECUTION WRAPPER ---------------------------------------------------------

def run_grimoire_pipeline(user_id: str, title: str, sql: str):
    """Headless execution entry point for the pipeline."""
    initial_state = {
        "user_id": user_id,
        "raw_title": title,
        "raw_sql": sql,
        "current_units": [],
        "processed_tools": [],
        "errors": [],
    }
    return grimoire_pipeline.invoke(initial_state)
