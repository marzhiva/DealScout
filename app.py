import streamlit as st
import os
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import requests
from typing import Literal

# Try to load environment variables from a .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Define the tools (copying from your notebook)
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
    st.error("GEMINI_API_KEY environment variable not set. Please set it in your system environment or a .env file.")
    st.stop()

# 1. Initialize the Gemini Model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=gemini_api_key).bind_tools(tools)

# 2. Node logic to invoke the model
def call_model(state: MessagesState):
    return {"messages": [model.invoke(state['messages'])]}

# New: Node logic for human approval for Streamlit
def human_approval(state: MessagesState):
    st.warning("--- Awaiting Human Approval for Destructive Operation ---")
    proposed_tool_calls = state['messages'][-1].tool_calls if hasattr(state['messages'][-1], 'tool_calls') else []

    if proposed_tool_calls:
        st.write(f"Agent proposed action: {proposed_tool_calls[0].args}")
        if st.session_state.get("approve_action", False):
             st.success("--- Human approval granted ---")
             st.session_state["approve_action"] = False # Reset approval
             return state
        else:
             st.session_state["pending_approval_state"] = state # Store state for re-execution after approval
             st.button("Approve Destructive Action", key="approve_btn", on_click=lambda: st.session_state.update(approve_action=True, rerun_app=True))
             st.stop()
    return state

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

st.title("DealScout: AI Commercial Real Estate Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_session"

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to safely parse out content from a message (fixes the concatenation error)
def get_message_content(message) -> str:
    if not hasattr(message, "content") or not message.content:
        return ""
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        text_parts = []
        for part in message.content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts)
    return str(message.content)

# Handle pending approval for Streamlit
if st.session_state.get("rerun_app", False):
    st.session_state["rerun_app"] = False
    if "pending_approval_state" in st.session_state:
        st.write("Executing approved action...")
        inputs = st.session_state["pending_approval_state"]
        st.session_state.messages.append({"role": "assistant", "content": "Approved, executing action..."})
        with st.chat_message("assistant"):
            st.markdown("Approved, executing action...")

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        try:
            for s in app.stream(inputs, config, stream_mode="values"):
                content = get_message_content(s["messages"][-1])
                if content:
                    st.session_state.messages.append({"role": "assistant", "content": content})
                    with st.chat_message("assistant"):
                        st.markdown(content)
            del st.session_state["pending_approval_state"]

        except Exception as e:
            st.error(f"An error occurred during approved action: {e}")


# React to user input
if prompt := st.chat_input("What would you like to do?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    inputs = {"messages": [("user", prompt)]}
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    try:
        full_response = ""
        # We track tools triggered in this stream block so we don't spam duplicate statements
        seen_tool_calls = set()

        for s in app.stream(inputs, config, stream_mode="values"):
            last_message = s["messages"][-1]
            
            # Safe string evaluation helper applied here
            msg_content = get_message_content(last_message)
            
            if msg_content and last_message.type == "ai":
                # With stream_mode="values", the last message updates progressively. 
                # Overwriting full_response prevents duplicating text chunks.
                full_response = msg_content
            elif hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tc in last_message.tool_calls:
                    tc_id = tc.get('id') if isinstance(tc, dict) else getattr(tc, 'id', None)
                    if tc_id and tc_id not in seen_tool_calls:
                        seen_tool_calls.add(tc_id)
                        name = tc.get('name') if isinstance(tc, dict) else tc.name
                        args = tc.get('args') if isinstance(tc, dict) else tc.args
                        full_response += f"\n*Agent decided to call tool: `{name}` with arguments {args}*\n"
            
            if st.session_state.get("pending_approval_state"):
                break

        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            with st.chat_message("assistant"):
                st.markdown(full_response)

    except Exception as e:
        st.error(f"An error occurred with the agent: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
