from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv
import os
import asyncio


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
# 3. BUILD GRAPH WITH MCP TOOLS
# =========================

async def build_graph():

    # Connect to running MCP server
    mcp_client = MultiServerMCPClient(
        {
            "arith": {
                "url": "http://localhost:8000",
                "transport": "streamable_http"
            }
        }
    )

    tools = await mcp_client.get_tools()

    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")

    # Conditional edge:
    graph.add_conditional_edges(
        "chat_node",
        tools_condition
    )

    graph.add_edge("tools", "chat_node")

    return graph.compile()


# =========================
# 4. RUN
# =========================

async def main():

    chatbot = await build_graph()

    result = await chatbot.ainvoke({
        "messages": [
            HumanMessage(
                content="Find the modulus of 132354 and 23 and give answer like a cricket commentator."
            )
        ]
    })

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())