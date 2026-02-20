from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from dotenv import load_dotenv
import os
import sqlite3
import requests


# =========================
# 1. ENV + LLM SETUP
# =========================

load_dotenv()

api_key = os.getenv("API_KEY1")

if not api_key:
    raise ValueError("❌ API_KEY1 not found in .env")

llm = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant"
)


# =========================
# 2. TOOLS
# =========================

# 🔎 Web Search Tool
duckduckgo = DuckDuckGoSearchRun(region="us-en")

@tool
def search(query: str) -> str:
    """Search the web for current information."""
    return duckduckgo.run(query)

search.name = "search"

# 🧮 Calculator Tool
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform arithmetic: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result
        }

    except Exception as e:
        return {"error": str(e)}


# 📈 Stock Price Tool (Hardcoded API key as requested)
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a symbol using Alpha Vantage.
    """
    url = (
        "https://www.alphavantage.co/query?"
        f"function=GLOBAL_QUOTE&symbol={symbol}&apikey=QRB59RORCKQ3BFK8"
    )

    response = requests.get(url)
    data = response.json()

    try:
        quote = data["Global Quote"]
        return {
            "symbol": symbol,
            "price": quote["05. price"],
            "latest_trading_day": quote["07. latest trading day"],
            "volume": quote["06. volume"]
        }
    except Exception:
        return {"error": "Invalid symbol or API limit reached", "raw": data}


tools = [search, calculator, get_stock_price]
llm_with_tools = llm.bind_tools(
    tools,
    tool_choice="auto"
)

# =========================
# 3. STATE
# =========================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# =========================
# 4. NODES
# =========================

from langchain_core.messages import SystemMessage

def chat_node(state: ChatState):
    messages = state["messages"]

    system = SystemMessage(
        content="You may only use these tools: search, calculator, get_stock_price. Do not call any other tool."
    )

    response = llm_with_tools.invoke([system] + messages)
    return {"messages": [response]}
tool_node = ToolNode(tools)


# =========================
# 5. CHECKPOINTER
# =========================

conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn=conn)


# =========================
# 6. GRAPH
# =========================

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges(
    "chat_node",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END
    }
)

graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# =========================
# 7. THREAD RETRIEVAL
# =========================

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
                "auto_named": False
            })

    return threads


print("successfully ran backend")