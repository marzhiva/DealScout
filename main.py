import os
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import requests

# Try to load environment variables from a .env file if it exists (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass # dotenv not installed, assume environment variables are set externally

# Define the tools (copying from your notebook, ensure these are available in your local environment)
class CalculateMetricsInput(BaseModel):
    """Input for calculating financial metrics."""
    noi: float = Field(..., description="Net Operating Income")
    purchase_price: float = Field(..., description="Purchase price of the property")

@tool("calculate_metrics", args_schema=CalculateMetricsInput)
def calculate_metrics(noi: float, purchase_price: float) -> str:
    """Calculates the Cap Rate for a property."""
    if purchase_price == 0:
        return "Error: Purchase price cannot be zero."
    cap_rate = (noi / purchase_price) * 100
    return f"The calculated Cap Rate is {cap_rate:.2f}%."

class SearchMarketDataInput(BaseModel):
    """Input for searching market data."""
    query: str = Field(..., description="The query to search for market data.")

@tool("search_market_data", args_schema=SearchMarketDataInput)
def search_market_data(query: str) -> str:
    """Searches for real estate market data using Serper API."""
    serper_api_key = os.environ.get("SERPER_API_KEY")
    if not serper_api_key:
        return "Error: SERPER_API_KEY not set in environment variables."

    url = "https://google.serper.dev/search"
    payload = {"q": query}
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.text
    else:
        return f"Error searching market data: {response.status_code} - {response.text}"

# Property watchlist and related tools
class PropertyWatchlist(BaseModel):
    watchlist: dict = Field(default_factory=dict, description="Dictionary of properties on the watchlist, keyed by property_id.")

property_watchlist = PropertyWatchlist()

class AddPropertyInput(BaseModel):
    property_id: str = Field(..., description="Unique identifier for the property.")
    address: str = Field(..., description="Address of the property.")
    noi: float = Field(..., description="Net Operating Income of the property.")
    purchase_price: float = Field(..., description="Purchase price of the property.")

@tool("add_property_to_watchlist", args_schema=AddPropertyInput)
def add_property_to_watchlist(property_id: str, address: str, noi: float, purchase_price: float) -> str:
    """Adds a property to the watchlist."""
    if property_id in property_watchlist.watchlist:
        return f"Error: Property ID {property_id} already exists in the watchlist."
    property_watchlist.watchlist[property_id] = {
        "address": address,
        "noi": noi,
        "purchase_price": purchase_price
    }
    return f"Property ID {property_id} ({address}) added to watchlist."

class DeletePropertyInput(BaseModel):
    property_id: str = Field(..., description="Unique identifier of the property to delete.")

@tool("delete_property_from_watchlist", args_schema=DeletePropertyInput)
def delete_property_from_watchlist(property_id: str) -> str:
    """Deletes a property from the watchlist after human approval."""
    if property_id not in property_watchlist.watchlist:
        return f"Error: Property ID {property_id} not found in the watchlist."
    del property_watchlist.watchlist[property_id]
    return f"Property ID {property_id} deleted from watchlist."

tools = [calculate_metrics, search_market_data, add_property_to_watchlist, delete_property_from_watchlist]

# Get API Key from environment variable
gemini_api_key = os.environ.get("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# 1. Initialize the Gemini Model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=gemini_api_key).bind_tools(tools)

# 2. Node logic to invoke the model
def call_model(state: MessagesState):
    return {"messages": [model.invoke(state['messages'])]}

# New: Node logic for human approval (simplified for CLI)
def human_approval(state: MessagesState):
    print("\n--- Awaiting Human Approval for Destructive Operation ---")
    proposed_tool_calls = state['messages'][-1].tool_calls if hasattr(state['messages'][-1], 'tool_calls') else []
    if proposed_tool_calls:
        first_tool_call_args = None
        if isinstance(proposed_tool_calls[0], dict):
            first_tool_call_args = proposed_tool_calls[0].get('args')
        elif hasattr(proposed_tool_calls[0], 'args'):
            first_tool_call_args = proposed_tool_calls[0].args

        if first_tool_call_args:
            print(f"Agent proposed: {first_tool_call_args}\n")

    # In a real CLI, you might prompt for input here:
    # approval = input("Do you approve this action? (yes/no): ").lower()
    # if approval == 'yes':
    print("--- Human approval granted (simulated for CLI) ---")
    return state

from typing import Literal
# 3. Construct the updated workflow Graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("human_approval", human_approval)

workflow.add_edge(START, "agent")

def route_agent_to_next(state: MessagesState) -> Literal["tools", "human_approval", "__end__"]:
    message = state['messages'][-1]
    if hasattr(message, 'tool_calls') and message.tool_calls:
        for tool_call in message.tool_calls:
            tool_name = None
            if isinstance(tool_call, dict):
                tool_name = tool_call.get('name')
            elif hasattr(tool_call, 'name'):
                tool_name = tool_call.name

            if tool_name == "delete_property_from_watchlist":
                return "human_approval"
        return "tools"
    return "__end__"

workflow.add_conditional_edges("agent", route_agent_to_next)
workflow.add_edge("human_approval", "tools")
workflow.add_edge("tools", "agent")

app = workflow.compile(checkpointer=MemorySaver())

def chat_cli():
    print("Welcome to DealScout CLI! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        inputs = {"messages": [("user", user_input)]}
        # Using a fixed thread_id for CLI for simplicity
        config = {"configurable": {"thread_id": "cli_session"}}

        try:
            for output in app.stream(inputs, config, stream_mode="values"):
                last_message = output["messages"][-1]
                # This is a simplified print. In a real CLI, you'd parse ToolMessage/AIMessage better
                if last_message.content:
                    print(f"Agent: {last_message.content}")
                elif last_message.tool_calls:
                    # For simplicity, print the tool calls made by the agent
                    print(f"Agent calls tool(s): {[(tc.get('name') if isinstance(tc, dict) else tc.name) for tc in last_message.tool_calls]}")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    chat_cli()
