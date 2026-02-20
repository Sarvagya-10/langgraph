import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph_database_backend import chatbot, retrieve_all_threads


# ---------- Utility functions ----------
def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in [t["id"] for t in st.session_state.chat_threads]:
        st.session_state.chat_threads.append({
            "id": thread_id,
            "name": thread_id[:8],
            "auto_named": False
        })


def reset_chat():
    st.session_state.thread_id = generate_thread_id()
    add_thread(st.session_state.thread_id)
    st.session_state.message_history = []


def delete_thread(thread_id):
    st.session_state.chat_threads = [
        t for t in st.session_state.chat_threads if t["id"] != thread_id
    ]

    if st.session_state.thread_id == thread_id:
        reset_chat()


def auto_rename_thread(thread_id, first_message):
    for t in st.session_state.chat_threads:
        if t["id"] == thread_id and not t["auto_named"]:
            t["name"] = first_message[:40]
            t["auto_named"] = True
            break


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ---------- Session setup ----------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = retrieve_all_threads()

existing_ids = {t["id"] for t in st.session_state.chat_threads}
if st.session_state.thread_id not in existing_ids:
    add_thread(st.session_state.thread_id)


# ---------- Sidebar ----------
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")

for thread in st.session_state.chat_threads[::-1]:
    thread_id = thread["id"]
    col1, col2 = st.sidebar.columns([6, 1])

    if col1.button(thread["name"], key=f"open_{thread_id}"):
        st.session_state.thread_id = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"

            temp_messages.append({
                "role": role,
                "content": msg.content
            })

        st.session_state.message_history = temp_messages
        st.rerun()

    if col2.button("🗑", key=f"del_{thread_id}"):
        delete_thread(thread_id)
        st.rerun()


# ---------- Display Previous Messages ----------
for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------- User Input ----------
user_input = st.chat_input("Type here...")

if user_input:

    # Save user message
    st.session_state.message_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    auto_rename_thread(st.session_state.thread_id, user_input)

    full_reply = ""

    # Assistant container
    with st.chat_message("assistant"):
        status_container = st.empty()
        response_container = st.empty()

        # Stream from LangGraph
        ##############################################
        for update in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": st.session_state.thread_id}},
            stream_mode="updates",
        ):

            for node, value in update.items():

                if node == "chat_node":
                    message = value["messages"][-1]

                    # Tool call
                    if isinstance(message, AIMessage) and message.tool_calls:
                        for tool_call in message.tool_calls:
                            tool_name = tool_call["name"]

                            if tool_name == "calculator":
                                status_container.info("🧮 Calculating...")
                            elif tool_name == "get_stock_price":
                                status_container.info("📈 Fetching stock price...")
                            elif tool_name == "duckduckgo_search":
                                status_container.info("🔍 Searching...")
                            else:
                                status_container.info(f"⚙️ Running {tool_name}...")

                    # Assistant streaming text
                    elif isinstance(message, AIMessage) and message.content:
                        full_reply += message.content
                        response_container.markdown(full_reply)

                elif node == "tools":
                    status_container.empty()
    # Save assistant reply once
    st.session_state.message_history.append({
        "role": "assistant",
        "content": full_reply
    })


print("successfully ran frontend")