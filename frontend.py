import streamlit as st
import requests
import uuid
import json
import os
from pypdf import PdfReader

# Use a live URL if deployed, otherwise fallback to localhost for testing
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Sentinel Research Pipeline", page_icon="🛡️", layout="wide")
st.title("🛡️ Sentinel Research Pipeline")

# --- CHAT MANAGEMENT HELPERS ---
CHATS_FILE = "chats.json"

def load_chats():
    if os.path.exists(CHATS_FILE):
        with open(CHATS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_chats(chats_dict):
    with open(CHATS_FILE, "w") as f:
        json.dump(chats_dict, f)

all_chats = load_chats()

# --- URL STATE PERSISTENCE ---
# Grab the thread_id from the browser URL, or generate a new one if it's a fresh visit
current_thread = st.query_params.get("thread_id")

if not current_thread:
    current_thread = str(uuid.uuid4())
    st.query_params["thread_id"] = current_thread

st.session_state.thread_id = current_thread

# --- FETCH HISTORY ON LOAD/SWITCH ---
# If this is a new session or we switched threads, pull the past data from the backend
if "chat_history" not in st.session_state or st.session_state.get("last_loaded_thread") != current_thread:
    try:
        res = requests.get(f"{BACKEND_URL}/history/{current_thread}")
        if res.status_code == 200:
            raw_history = res.json().get("chat_history", [])
            parsed_history = []
            
            # Convert backend array ["User: hi", "Sentinel: hello"] to Streamlit dictionaries
            for msg in raw_history:
                if msg.startswith("User: "):
                    parsed_history.append({"role": "user", "content": msg.replace("User: ", "", 1)})
                elif msg.startswith("Sentinel: "):
                    parsed_history.append({"role": "assistant", "content": msg.replace("Sentinel: ", "", 1)})
            
            st.session_state.chat_history = parsed_history
        else:
            st.session_state.chat_history = []
    except Exception:
        st.session_state.chat_history = []
        
    st.session_state.last_loaded_thread = current_thread
    st.session_state.agent_status = "idle"
    st.session_state.draft_text = ""

# --- SIDEBAR INTERFACE ---
with st.sidebar:
    st.header("💬 Conversations")
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_thread = str(uuid.uuid4())
        st.query_params["thread_id"] = new_thread
        st.session_state.chat_history = []
        st.session_state.agent_status = "idle"
        st.rerun()

    # List Past Chats
    st.subheader("Past Chats")
    for tid, title in reversed(all_chats.items()):
        if st.button(title, key=tid, use_container_width=True):
            st.query_params["thread_id"] = tid
            st.rerun()

    st.divider()
    
    st.header("📁 Document Ingestion")
    uploaded_file_context = ""
    uploaded_file = st.file_uploader("Upload a document (.txt or .pdf)", type=["txt", "pdf"])
    
    if uploaded_file is not None:
        st.success(f"Loaded: {uploaded_file.name}")
        if uploaded_file.type == "text/plain":
            uploaded_file_context = str(uploaded_file.read().decode("utf-8"))
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            uploaded_file_context = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

# --- RENDER CHAT HISTORY ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- PROCESS NEW MESSAGES ---
# --- PROCESS NEW MESSAGES ---
if user_query := st.chat_input("Ask Sentinel anything...", disabled=(st.session_state.agent_status != "idle")):
    
    # Save to sidebar list if it's the first message
    if current_thread not in all_chats:
        all_chats[current_thread] = user_query[:25] + "..."
        save_chats(all_chats)
        # ❌ REMOVED st.rerun() FROM HERE! We don't want to interrupt the script!
        
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
        
    st.session_state.agent_status = "running"
    
    # ... (the rest of the block stays exactly the same)
    
    with st.status("Executing Agent Loop...", expanded=True) as status_box:
        try:
            payload = {
                "query": user_query, 
                "thread_id": current_thread,
                "file_context": uploaded_file_context 
            }
            
            # Ensure no double slashes in the URL string
            clean_url = BACKEND_URL.rstrip('/')
            
            # --- NEW: Check for server errors (like 502 or 404) before parsing ---
            response = requests.post(f"{clean_url}/stream", json=payload, stream=True)
            
            if response.status_code != 200:
                st.error(f"Backend Server Error ({response.status_code}): {response.text}")
                st.session_state.agent_status = "idle"
                st.stop()
                
            for line in response.iter_lines():
                if line:
                    data_str = line.decode('utf-8').replace("data: ", "")
                    try:
                        update = json.loads(data_str)
                        
                        if "error" in update:
                            st.error(f"Backend Error: {update['error']}")
                            st.session_state.agent_status = "idle"
                            st.stop()
                            
                        for node_name, state_data in update.items():
                            if node_name == "__interrupt__":
                                st.session_state.agent_status = "awaiting_approval"
                                status_box.update(label="Awaiting Review!", state="running")
                                st.rerun()
                                
                            st.write(f"✅ **{node_name.upper()}** completed.")
                            
                            # Only update the draft text if it actually exists in the dictionary
                            if isinstance(state_data, dict) and "current_draft" in state_data:
                                st.session_state.draft_text = state_data["current_draft"]
                            
                            if node_name == "chitchat":
                                if st.session_state.draft_text.strip():
                                    st.session_state.chat_history.append({"role": "assistant", "content": st.session_state.draft_text})
                                st.session_state.agent_status = "idle"
                                st.rerun()
                                
                    except json.JSONDecodeError:
                        pass
                        
            if st.session_state.agent_status == "running":
                if st.session_state.draft_text.strip():
                    st.session_state.chat_history.append({"role": "assistant", "content": st.session_state.draft_text})
                st.session_state.agent_status = "idle"
                st.rerun()
                
        except Exception as e:
            st.error(f"Connection Failed: {str(e)}")
            st.session_state.agent_status = "idle"

# --- HUMAN-IN-THE-LOOP PANEL ---
if st.session_state.agent_status == "awaiting_approval":
    st.markdown("---")
    st.subheader("🤖 Sentinel Draft Review")
    
    edited_draft = st.text_area("Review/Edit generated draft:", value=st.session_state.draft_text, height=250)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Approve & Finalize", use_container_width=True, type="primary"):
            with st.spinner("Finalizing..."):
                res = requests.post(
                    f"{BACKEND_URL}/resume",
                    json={"thread_id": current_thread, "action": "approve", "edited_draft": edited_draft}
                )
                final_report = res.json().get("final_report", edited_draft)
                st.session_state.chat_history.append({"role": "assistant", "content": final_report})
                st.session_state.agent_status = "idle"
                st.rerun()
                
    with col2:
        with st.popover("❌ Reject & Redirect Agent", use_container_width=True):
            human_feedback = st.text_area("Provide steering feedback:")
            if st.button("Force Recalculation", type="primary"):
                with st.spinner("Rerouting Agent..."):
                    res = requests.post(
                        f"{BACKEND_URL}/resume",
                        json={"thread_id": current_thread, "action": "reject", "edited_draft": human_feedback}
                    )
                    st.session_state.draft_text = res.json().get("final_report", "")
                    st.session_state.agent_status = "awaiting_approval"
                    st.rerun()
