import streamlit as st
import uuid
import time
from langchain_core.messages import HumanMessage
from langgraph_backend import chatbot


# ---------- Session setup ----------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "message_history" not in st.session_state:
    st.session_state.message_history = []


# ---------- Load previous messages ----------
for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------- User input ----------
user_input = st.chat_input("Type here...")

if user_input:
    # 1️⃣ Store & display user message
    st.session_state.message_history.append({
        "role": "user",
        "content": user_input
    })
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2️⃣ Convert history → LangGraph messages
    messages = [
        HumanMessage(content=m["content"])
        for m in st.session_state.message_history
        if m["role"] == "user"
    ]

    # 3️⃣ Stream assistant reply with typewriter effect
    response_container = st.empty()
    full_reply = ""

    for chunk, metadata in chatbot.stream(
        {"messages": messages},
        config={"configurable": {"thread_id": st.session_state.thread_id}},
        stream_mode="messages",
    ):
        if chunk.content:
            for char in chunk.content:
                full_reply += char
                response_container.chat_message("assistant").markdown(full_reply)
                time.sleep(0.001)

    # 4️⃣ Save assistant reply
    st.session_state.message_history.append({
        "role": "assistant",
        "content": full_reply
    })
