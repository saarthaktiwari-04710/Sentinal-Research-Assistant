import os
from typing import TypedDict, Annotated, List, Any
import operator
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from tools import local_filesystem_search, web_search_fallback

# --- STATE DEFINITION ---
def overwrite(existing: Any, new: Any) -> Any:
    return new

class SentinelState(TypedDict):
    original_query: str
    chat_history: Annotated[List[str], operator.add]  # Memory system
    current_sub_question: Annotated[str, overwrite]
    retrieved_context: Annotated[str, overwrite]
    current_draft: Annotated[str, overwrite]
    critic_feedback: Annotated[str, overwrite]
    approved_answers: Annotated[List[str], operator.add]
    loop_count: int

# --- LLM SETUP ---
# Using the supported model we updated earlier
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)

# --- SCHEMAS ---
class GraderOutput(BaseModel):
    relevance: str = Field(description="'RELEVANT' or 'IRRELEVANT'")

structured_grader = llm.with_structured_output(GraderOutput)

class CriticOutput(BaseModel):
    status: str = Field(description="'APPROVED' or 'REJECTED'")
    feedback: str = Field(description="Reason for rejection, if any.")

structured_critic = llm.with_structured_output(CriticOutput)

class RouteOutput(BaseModel):
    intent: str = Field(description="'CHITCHAT' or 'RESEARCH'")

structured_router = llm.with_structured_output(RouteOutput)

# --- CORE RESEARCH NODES ---
async def planner_node(state: SentinelState):
    """Contextualizes the search query using previous memory."""
    history = "\n".join(state.get("chat_history", [])[:-1]) # Get history excluding the current prompt
    
    # If there is no prior conversation, just use the original query
    if not history.strip():
        return {"current_sub_question": state["original_query"], "loop_count": state.get("loop_count", 0)}
        
    # Otherwise, have the LLM rewrite the query so it makes sense on its own
    rewrite_prompt = (
        f"Here is the conversation so far:\n{history}\n\n"
        f"The user just asked: '{state['original_query']}'. \n"
        f"Rewrite this as a standalone search query that includes all necessary context from the conversation. "
        f"Return ONLY the new search query."
    )
    
    response = await llm.ainvoke(rewrite_prompt)
    
    return {"current_sub_question": response.content, "loop_count": state.get("loop_count", 0)}

async def retriever_node(state: SentinelState):
    """Fetches local documents and combines them with uploaded file text."""
    # Fetch content using your original filesystem search tool
    system_context = await local_filesystem_search.ainvoke(state["current_sub_question"])
    
    # Check if the frontend passed down uploaded file contents
    uploaded_context = state.get("retrieved_context", "")
    
    # Merge both context pools seamlessly
    combined_context = ""
    if uploaded_context.strip():
        combined_context += f"[UPLOADED DOCUMENT CONTEXT]:\n{uploaded_context}\n\n"
    if "No local documents found" not in system_context:
        combined_context += f"[SYSTEM DATABASE CONTEXT]:\n{system_context}"
        
    if not combined_context.strip():
        combined_context = "No local documents found regarding this topic."
        
    return {"retrieved_context": combined_context}

async def crag_grader_node(state: SentinelState):
    """Evaluates if the context is relevant. Falls back to web if not."""
    grade = await structured_grader.ainvoke(
        f"Question: {state['current_sub_question']}\nContext: {state['retrieved_context']}\nIs this context relevant?"
    )
    
    if grade.relevance == "IRRELEVANT":
        # Fallback to live internet search via DuckDuckGo
        new_context = await web_search_fallback.ainvoke(state["current_sub_question"])
        return {"retrieved_context": new_context}
    return {}

async def generator_node(state: SentinelState):
    """Drafts the answer using the verified context."""
    prompt = f"Answer '{state['current_sub_question']}' using ONLY this context: {state['retrieved_context']}"
    draft = await llm.ainvoke(prompt)
    return {"current_draft": draft.content}

async def critic_node(state: SentinelState):
    """Self-RAG check to prevent hallucinations."""
    evaluation = await structured_critic.ainvoke(
        f"Draft: {state['current_draft']}\nContext: {state['retrieved_context']}\nDoes the draft hallucinate outside the context?"
    )
    new_loop_count = state["loop_count"] + 1
    return {
        "critic_feedback": evaluation.status, 
        "loop_count": new_loop_count,
        "approved_answers": [state["current_draft"]] if evaluation.status == "APPROVED" else []
    }

async def synthesizer_node(state: SentinelState):
    """Commits the human-approved draft to memory without rewriting it."""
    
    # Grab the draft that the human just approved or edited in the UI
    final_report = state["current_draft"]
    
    # Simply save it to the chat history so the agent remembers it, 
    # without running it through the LLM again!
    return {
        "chat_history": [f"Sentinel: {final_report}"]
    }

# --- CHITCHAT & ROUTING NODES ---
async def chitchat_node(state: SentinelState):
    """Handles casual greetings using conversation memory."""
    history_list = state.get("chat_history", [])
    history = "\n".join(history_list)
    
    # --- DEBUG PRINT: Check your Uvicorn terminal to see this! ---
    print("\n--- DEBUG: CHITCHAT NODE MEMORY LOOKUP ---")
    print(f"Current Thread History count: {len(history_list)} messages")
    print(history)
    print("------------------------------------------\n")
    
    # We use a structured system prompt to force the model to look at the history log
    prompt = (
        f"System: You are Sentinel, an AI assistant. You must use the conversation history provided below "
        f"to remember context like the user's name, preferences, or previous statements.\n\n"
        f"Conversation History:\n{history}\n\n"
        f"User's Latest Message: '{state['original_query']}'\n\n"
        f"Response:"
    )
    
    response = await llm.ainvoke(prompt)
    return {
        "current_draft": response.content,
        "chat_history": [f"Sentinel: {response.content}"]
    }

async def route_initial_query(state: SentinelState):
    """Determines intent while being aware of the conversation history and handling schema errors."""
    history = "\n".join(state.get("chat_history", []))
    
    # Check if a document was uploaded
    has_document = "Yes" if state.get("retrieved_context") else "No"
    
    prompt = (
        f"Review the conversation history:\n{history}\n\n"
        f"Did the user upload a document for this query? {has_document}\n\n"
        f"Classify the user's latest query: '{state['original_query']}'.\n"
        f"Rules:\n"
        f"- If the user greets you, says goodbye, or asks personal questions about you or themselves, choose 'CHITCHAT'.\n"
        f"- If the user asks you to explain something, summarize a document, solve a problem, write code, or asks a question requiring facts/data, choose 'RESEARCH'.\n"
        f"- If a document was uploaded, ALWAYS choose 'RESEARCH'."
    )
    
    try:
        route = await structured_router.ainvoke(prompt)
        
        if route.intent == "CHITCHAT":
            return "chitchat"
        return "planner"
        
    except Exception as e:
        # If the LLM hallucinates the JSON schema (like outputting "name" instead of "intent"),
        # we catch the error, print it to the terminal for debugging, and safely default to the RESEARCH pipeline.
        print(f"\n--- ROUTER SCHEMA ERROR HANDLED --- \n{str(e)}\nDefaulting to 'RESEARCH' path.\n")
        return "planner"

def route_critic(state: SentinelState):
    """Routes based on the critic's approval."""
    if state["critic_feedback"] == "APPROVED":
        return "synthesizer"
    if state["loop_count"] >= 3:
        return "synthesizer" # Circuit breaker
    return "retriever"

# --- BUILD THE GRAPH ---
builder = StateGraph(SentinelState)

# Add all nodes
builder.add_node("planner", planner_node)
builder.add_node("retriever", retriever_node)
builder.add_node("crag_grader", crag_grader_node)
builder.add_node("generator", generator_node)
builder.add_node("critic", critic_node)
builder.add_node("synthesizer", synthesizer_node)
builder.add_node("chitchat", chitchat_node)

# Map the edges
builder.add_conditional_edges(START, route_initial_query, {"chitchat": "chitchat", "planner": "planner"})
builder.add_edge("planner", "retriever")
builder.add_edge("retriever", "crag_grader")
builder.add_edge("crag_grader", "generator")
builder.add_edge("generator", "critic")
builder.add_conditional_edges("critic", route_critic, {"synthesizer": "synthesizer", "retriever": "retriever"})
builder.add_edge("synthesizer", END)
builder.add_edge("chitchat", END)
