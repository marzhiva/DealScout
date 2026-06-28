# app.py
import streamlit as st
from langchain_core.messages import HumanMessage
# IMPORT THE APP GRAPH OBJECT HERE TOO:
from agent_core import app 

st.set_page_config(page_title="DealScout AI Agent", page_icon="🏢")
st.title("DealScout: AI Commercial Real Estate Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI Acquisitions Agent. How can I help you analyze deals today?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_query := st.chat_input("Ask DealScout..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    config = {"configurable": {"thread_id": "streamlit_session"}}
    response = app.invoke({"messages": [HumanMessage(content=user_query)]}, config)
    agent_reply = response["messages"][-1].content

    with st.chat_message("assistant"):
        st.write(agent_reply)
    st.session_state.messages.append({"role": "assistant", "content": agent_reply})
