import streamlit as st
import uuid

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []



#load conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('type here..')

from langchain_core.messages import HumanMessage
from langgraph_backend import chatbot

if user_input:
    # 1. Show & store user message
    st.session_state['message_history'].append({
        'role': 'user',
        'content': user_input
    })
    with st.chat_message('user'):
        st.text(user_input)

    # 2. Convert history → LangGraph format
    messages = [
        HumanMessage(content=m["content"])
        for m in st.session_state['message_history']
        if m["role"] == "user"
    ]

    # 3. Call LangGraph
    result = chatbot.invoke(
    {"messages": messages},
    config={"configurable": {"thread_id": st.session_state.thread_id}})

    assistant_reply = result["messages"][-1].content

    # 4. Store & show assistant reply
    st.session_state['message_history'].append({
        'role': 'assistant',
        'content': assistant_reply
    })
    with st.chat_message('assistant'):
        st.text(assistant_reply)
