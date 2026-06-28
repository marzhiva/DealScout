# agent_core.py
import os
import json
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# =====================================================================
# TOOLS DEFINITION
# =====================================================================
@tool
def calculate_metrics(noi: float, purchase_price: float) -> str:
    """Calculates the Cap Rate and basic financial stability of a commercial property."""
    if purchase_price <= 0:
        return "Error: Purchase price must be greater than zero."
    cap_rate = (noi / purchase_price) * 100
    return f"The calculated Cap Rate is {cap_rate:.2f}%."

@tool
def search_market_data(query: str) -> str:
    """Fetches real-time external real estate market trends using the Serper API."""
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': os.environ.get("SERPER_API_KEY", ""),
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(url, headers=headers, data=payload)
        results = response.json()
        return results.get('organic', [{}])[0].get('snippet', 'No data found.')
    except Exception as e:
        return f"Failed to fetch live market data: {str(e)}"

tools = [calculate_metrics, search_market_data]

# =====================================================================
# GRAPH COMPILATION
# =====================================================================
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0).bind_tools(tools)

def call_model(state: MessagesState):
    return {"messages": [model.invoke(state['messages'])]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

# This is what gets exported
app = workflow.compile(checkpointer=MemorySaver())
