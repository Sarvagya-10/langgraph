from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from groq import Groq
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
import sqlite3


# Load variables from .env into environment
load_dotenv()

# Get the key
api_key = os.getenv("API_KEY1")

# Ensure the key exists
if not api_key:
    raise ValueError("❌ API_KEY not found in .env file")


client = Groq(api_key=api_key)


llm = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant"
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
    
    
def chat_node(state: ChatState):
    messages = state['messages']
    
    response = llm.invoke(messages)
    
    return {'messages':[response]}



conn = sqlite3.connect(database = 'chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)


chatbot = graph.compile(checkpointer=checkpointer)
chatbot


def retrieve_all_threads():
    seen = set()
    threads = []

    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]

        if thread_id not in seen:
            seen.add(thread_id)
            threads.append({
                "id": thread_id,
                "name": thread_id[:8],
                "auto_named": False,
            })

    return threads

    