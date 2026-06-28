import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="DealScout AI Agent", page_icon="🏢")
st.title("DealScout: AI Commercial Real Estate Assistant")

# Initialize Chat Sessions Memory inside Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI Acquisitions Agent. How can I help you analyze deals today?"}]

# Render previous history bubbles
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Capture user interaction
if user_query := st.chat_input("Ask DealScout to calculate metrics or search market data..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Invoke your Compiled Graph here 
    # (Make sure 'app' graph compiled from LangGraph is imported or defined here)
    config = {"configurable": {"thread_id": "streamlit_session"}}
    response = app.invoke({"messages": [HumanMessage(content=user_query)]}, config)
    agent_reply = response["messages"][-1].content

    # Render Assistant bubble
    with st.chat_message("assistant"):
        st.write(agent_reply)
    st.session_state.messages.append({"role": "assistant", "content": agent_reply})