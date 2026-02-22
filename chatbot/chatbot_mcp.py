from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient


# =========================
# 1. ENV + LLM SETUP
# =========================

load_dotenv()

api_key = os.getenv("API_KEY1")

if not api_key:
    raise ValueError("API_KEY1 not found in .env")

llm = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant"
)


# =========================
# 2. STATE
# =========================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# =========================
# 3. GRAPH BUILDER
# =========================

def build_graph():

    async def chat_node(state: ChatState):
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)

    return graph.compile()


# =========================
# 4. RUN
# =========================

async def main():
    chatbot = build_graph()

    result = await chatbot.ainvoke({
        "messages": [
            HumanMessage(content="Find the modulus of 132354 and 23 and give answer like a cricket commentator.")
        ]
    })

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())