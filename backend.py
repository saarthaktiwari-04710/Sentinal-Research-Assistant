import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from graph import builder  # Ensures graph.py is in the same folder

# Load environment variables
load_dotenv()

# Define global variable for the agent
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    
    # Using AsyncSqliteSaver for zero-config local persistence
    async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
        await checkpointer.setup() # Automatically creates local database tables
        
        # Compile the graph with persistence and a HITL interrupt BEFORE the synthesizer
        agent = builder.compile(
            checkpointer=checkpointer,
            interrupt_before=["synthesizer"]
        )
        yield # Application runs here

# --- THIS IS THE LINE UVICORN WAS MISSING ---
app = FastAPI(lifespan=lifespan)

class RunRequest(BaseModel):
    query: str
    thread_id: str
    file_context: str = ""  # NEW: Holds the raw text extracted from the document

class ResumeRequest(BaseModel):
    thread_id: str
    action: str 
    edited_draft: str = None

@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    """Fetches the past chat history safely from the SQLite checkpointer."""
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # 1. Fetch the latest configuration checkpoint state
        state = await agent.aget_state(config)
        
        # 2. Extract values safely if they exist
        if state and state.values:
            history = state.values.get("chat_history", [])
            return {"chat_history": history}
            
        # 3. Fallback: If aget_state returns empty, iterate through state history 
        # to find the most recent checkpoint containing the chat log
        async for state_update in agent.aget_state_history(config):
            if state_update.values and "chat_history" in state_update.values:
                history = state_update.values["chat_history"]
                if history:
                    return {"chat_history": history}
                    
    except Exception as e:
        print(f"Error fetching history: {str(e)}")
        
    return {"chat_history": []}


@app.post("/stream")
async def stream_agent(request: RunRequest):
    async def sse_generator():
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # 1. Safely fetch the existing history for this thread
        existing_state = await agent.aget_state(config)
        current_history = []
        if existing_state and hasattr(existing_state, 'values') and "chat_history" in existing_state.values:
            current_history = existing_state.values["chat_history"]
            
        # 2. Append the new message without deleting the old ones
        current_history.append(f"User: {request.query}")
        
        # 3. Pass the full, combined history to the graph
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
    """Resumes the graph based on human approval or rejection."""
    config = {"configurable": {"thread_id": request.thread_id}}
    
    if request.action == "approve":
        if request.edited_draft:
            # Save any manual typos the human fixed
            await agent.aupdate_state(config, {"current_draft": request.edited_draft})
            
        # Resume normally -> Goes to Synthesizer
        async for chunk in agent.astream(None, config, stream_mode="updates"):
             pass 
             
    elif request.action == "reject":
        # Mutate the state with the human's instructions
        new_query = f"HUMAN CORRECTION: {request.edited_draft}"
        
        # We update the state AS the 'critic' node, telling LangGraph it was rejected.
        # This triggers the conditional edge to route back to the Retriever!
        await agent.aupdate_state(
            config, 
            {
                "current_sub_question": new_query, 
                "critic_feedback": "REJECTED",
                "loop_count": 0 # Give it 3 fresh tries
            },
            as_node="critic" 
        )
        
        # Resume the graph (it will now travel backwards to the Retriever)
        async for chunk in agent.astream(None, config, stream_mode="updates"):
             pass 
         
    final_state = await agent.aget_state(config)
    return {"final_report": final_state.values.get("current_draft")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
