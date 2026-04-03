# from typing import TypedDict, List, Optional, Annotated
# import json
# from langgraph.graph import StateGraph, END
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage, SystemMessage

# # Import your existing prompts and registry logic
# from .prompts import build_stage1_prompt, build_stage2_prompt
# from .registry import registry_check_tool, save_final_tool
# # from .embedder import embed_tool_description # Import when ready

# # --- 1. DEFINE THE SHARED STATE ----------------------------------------------
# # This object is passed between every node in the graph.

# class GrimoireState(TypedDict):
#     # Input Data
#     user_id: str
#     raw_title: str
#     raw_sql: str
    
#     # Discovery Context (Loaded once)
#     domains: List[str]
#     category_map: dict
#     registry_context: str # Minified JSON of existing tools for matching
    
#     # Pipeline Artifacts
#     current_units: List[dict] # Output from Stage 1 (Decomposition)
#     processed_tools: List[dict] # Output from Stage 2 (Classification)
    
#     # Error Handling / Validation
#     errors: List[str]
#     retry_count: int

# # --- 2. DEFINE THE NODES (TRANSFORMATION STEPS) -------------------------------
# # Each function takes the current state and returns an updated state.

# # Initialize the LLM (Use GPT-4 for high-reasoning taxonomy tasks)
# llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# def initialize_context_node(state: GrimoireState) -> GrimoireState:
#     """Load taxonomy and registry context from Postgres."""
#     print("--- [Node: Initialize Context] ---")
#     # TODO: Replace placeholders with real DB calls from .registry
#     domains = ["Protocols", "Tokens", "Chains"]
#     category_map = {
#         "Protocols": ["Yield Analytics", "Lending Activity"],
#         "Tokens": ["Price Analytics", "Supply Analytics"]
#     }
#     # A minified snapshot of existing g3/inputs for matching
#     registry_json = '[{"id": "t1", "g3": "Supply Tracker", "inputs": ["chain", "protocol"]}]'
    
#     return {**state, "domains": domains, "category_map": category_map, "registry_context": registry_json, "errors": [], "retry_count": 0, "processed_tools": []}

# def stage1_decompose_node(state: GrimoireState) -> GrimoireState:
#     """Decompose raw SQL into logical units."""
#     print("--- [Node: Stage 1 - Decompose] ---")
    
#     prompt_text = build_stage1_prompt(state['raw_title'], state['raw_sql'])
#     messages = [SystemMessage(content="You are an expert SQL decompiler."), HumanMessage(content=prompt_text)]
    
#     response = llm.invoke(messages)
    
#     try:
#         # Strict validation: Ensure output is valid JSON
#         cleaned_response = response.content.replace("```json", "").replace("```", "").strip()
#         data = json.loads(cleaned_response)
#         units = data.get("units", [])
#         return {**state, "current_units": units}
#     except json.JSONDecodeError:
#         error_msg = f"Stage 1 failed to generate valid JSON: {response.content[:100]}..."
#         return {**state, "errors": [error_msg]}

# def stage2_classify_node(state: GrimoireState) -> GrimoireState:
#     """Classify each unit into the taxonomy and extract inputs."""
#     print(f"--- [Node: Stage 2 - Classify {len(state['current_units'])} units] ---")
    
#     classified_tools = []
#     errors = []
    
#     for unit in state['current_units']:
#         prompt_text = build_stage2_prompt(
#             label=unit['label'],
#             sql=unit['sql'],
#             registry_json=state['registry_context'],
#             domains=state['domains'],
#             category_map=state['category_map']
#         )
        
#         messages = [SystemMessage(content="You are an expert onchain taxonomy engineer."), HumanMessage(content=prompt_text)]
#         response = llm.invoke(messages)
        
#         try:
#             cleaned_response = response.content.replace("```json", "").replace("```", "").strip()
#             tool_data = json.loads(cleaned_response)
            
#             # Inject the original SQL back into the tool data
#             tool_data['sql_code'] = unit['sql']
#             classified_tools.append(tool_data)
            
#         except json.JSONDecodeError:
#             errors.append(f"Stage 2 failed for unit '{unit['label']}': {response.content[:100]}...")

#     return {**state, "processed_tools": classified_tools, "errors": errors}

# def registry_match_node(state: GrimoireState) -> GrimoireState:
#     """Check processed tools against the Postgres registry (Idempotency)."""
#     print("--- [Node: Registry Match] ---")
    
#     final_tools_to_save = []
    
#     for tool in state['processed_tools']:
#         # Call .registry to check if g3 + inputs already exists
#         # This is crucial for maintaining a clean, de-duplicated Grimoire.
#         match_result = registry_check_tool(tool, state['registry_context'])
        
#         if match_result['match']:
#             # Tool exists, we just need the ID. We don't save a new version.
#             tool['id'] = match_result['tool_id']
#             tool['note'] = "Matched existing tool"
#         else:
#             # New tool or input extension needed.
#             tool['id'] = None # Will be generated on save
            
#         final_tools_to_save.append(tool)
        
#     return {**state, "processed_tools": final_tools_to_save}

# def finalize_tool_node(state: GrimoireState) -> GrimoireState:
#     """Save new tools to Postgres and generate vector embeddings."""
#     print("--- [Node: Finalize & Save] ---")
    
#     saved_ids = []
#     for tool in state['processed_tools']:
#         if tool['id'] is None:
#             # 1. Save metadata and SQL to Postgres relational tables
#             new_id = save_final_tool(tool, state['user_id'])
            
#             # 2. TODO: Generate vector embedding of g3+g4 and store to pgvector
#             # description = f"{tool['g3']} - {tool['g4']}"
#             # embed_tool_description(new_id, description)
            
#             saved_ids.append(new_id)
#         else:
#             saved_ids.append(tool['id']) # Already existed
            
#     print(f"Pipeline complete. {len(saved_ids)} tools processed.")
#     return state # Final state

# # --- 3. DEFINE THE CONDITIONAL LOGIC (ROUTING) -------------------------------

# def should_continue(state: GrimoireState):
#     """Router: Checks for critical errors or retry limits."""
#     if state['errors']:
#         print(f"!!! Pipeline Error Detected: {state['errors'][0]}")
#         return "end" # Stop the pipeline if validation fails
    
#     if state['retry_count'] > 3:
#         print("!!! Max Retries Exceeded.")
#         return "end"
        
#     return "continue"

# # --- 4. CONSTRUCT THE GRAPH (THE ORCHESTRATION DAG) ----------------------------

# workflow = StateGraph(GrimoireState)

# # A. Add Nodes
# workflow.add_node("initialize", initialize_context_node)
# workflow.add_node("stage1_decompose", stage1_decompose_node)
# workflow.add_node("stage2_classify", stage2_classify_node)
# workflow.add_node("registry_match", registry_match_node)
# workflow.add_node("finalize", finalize_tool_node)

# # B. Set Entry Point
# workflow.set_entry_point("initialize")

# # C. Define Edges (Linear flow)
# workflow.add_edge("initialize", "stage1_decompose")
# workflow.add_edge("stage1_decompose", "stage2_classify")
# workflow.add_edge("stage2_classify", "registry_match")
# workflow.add_edge("registry_match", "finalize")

# # D. Set End Point
# workflow.add_edge("finalize", END)

# # E. Compile the Graph
# grimoire_pipeline = workflow.compile()

# # --- 5. EXECUTION WRAPPER ---------------------------------------------------

# def run_grimoire_pipeline(user_id: str, title: str, sql: str):
#     """Headless execution entry point for the pipeline."""
#     initial_state = {
#         "user_id": user_id,
#         "raw_title": title,
#         "raw_sql": sql,
#         "current_units": [],
#         "processed_tools": [],
#         "errors": []
#     }
    
#     # Execute the graph
#     final_output = grimoire_pipeline.invoke(initial_state)
#     return final_output