import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# IMPORTANT: Import your StateGraph builder from your graph.py file here.
# For example, if your graph builder object is named 'workflow' in graph.py:
from graph import builder

# --- CONFIGURATION ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is missing!")

# Global agent variable to be initialized in lifespan
agent = None 

# --- LIFESPAN (DATABASE CONNECTION) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    
    # Connect to the permanent Neon PostgreSQL database
    async with AsyncConnectionPool(conninfo=DATABASE_URL) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        
        # Automatically creates the memory tables (checkpoints, writes, etc.) in Neon
        await checkpointer.setup()
        
        # Compile your custom graph with the Postgres checkpointer attached
        agent = workflow.compile(checkpointer=checkpointer)
        
        yield # The FastAPI server runs while paused here

# Initialize FastAPI
app = FastAPI(lifespan=lifespan)

# --- PYDANTIC MODELS ---
class RunRequest(BaseModel):
    query: str
    thread_id: str
    file_context: str = ""

class ResumeRequest(BaseModel):
    thread_id: str
    action: str
    edited_draft: str

# --- ENDPOINTS ---

@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    """Fetches the past chat history safely from the Postgres checkpointer."""
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # 1. Fetch the latest configuration checkpoint state
        state = await agent.aget_state(config)
        
        # 2. Extract values safely if they exist
        if state and hasattr(state, 'values') and state.values:
            history = state.values.get("chat_history", [])
            return {"chat_history": history}
            
        # 3. Fallback: Iterate through state history to find the most recent chat log
        async for state_update in agent.aget_state_history(config):
            if hasattr(state_update, 'values') and "chat_history" in state_update.values:
                history = state_update.values["chat_history"]
                if history:
                    return {"chat_history": history}
                    
    except Exception as e:
        print(f"Error fetching history: {str(e)}")
        
    return {"chat_history": []}


@app.post("/stream")
async def stream_agent(request: RunRequest):
    """Executes the graph and streams the state updates back to the frontend."""
    async def sse_generator():
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # 1. Safely fetch the existing history for this thread
        existing_state = await agent.aget_state(config)
        current_history = []
        
        if existing_state and hasattr(existing_state, 'values') and "chat_history" in existing_state.values:
            # Create a copy of the list so we don't mutate the reference
            current_history = list(existing_state.values["chat_history"])
            
        # 2. Append the new message without deleting the old ones
        current_history.append(f"User: {request.query}")
        
        # 3. Pass the full, combined history to the graph inputs
        inputs = {
            "original_query": request.query, 
            "chat_history": current_history, 
            "retrieved_context": request.file_context if request.file_context else "",
            "loop_count": 0
        }
        
        try:
            async for chunk in agent.astream(inputs, config, stream_mode="updates"):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            error_data = {"error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
            
    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@app.post("/resume")
async def resume_agent(request: ResumeRequest):
    """Handles the human-in-the-loop approval or rejection of drafts."""
    config = {"configurable": {"thread_id": request.thread_id}}
    
    try:
        # Update the graph state with the human's input as an interrupt payload
        await agent.aupdate_state(
            config, 
            {"current_draft": request.edited_draft, "human_feedback": request.action}
        )
        
        # Resume the graph execution (passing None for inputs means "resume from interrupt")
        async for chunk in agent.astream(None, config, stream_mode="updates"):
            pass 
            
        # Fetch the very latest state to return the finalized report
        state = await agent.aget_state(config)
        return {"final_report": state.values.get("current_draft", request.edited_draft)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
