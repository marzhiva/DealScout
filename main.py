# main.py
import os
# IMPORT THE APP GRAPH OBJECT FROM YOUR CORE FILE HERE:
from agent_core import app 

print("--- DealScout AI Agent Terminal Interface ---")
print("Type 'exit' or 'quit' to stop the loop.\n")

config = {"configurable": {"thread_id": "cli_session"}}

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    
    if not user_input.strip():
        continue
        
    inputs = {"messages": [{"role": "user", "content": user_input}]}
    
    final_msg = None
    for output in app.stream(inputs, config, stream_mode="values"):
        final_msg = output["messages"][-1]
    
    if final_msg:
        print(f"Agent: {final_msg.content}\n")
