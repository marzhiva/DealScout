import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Ensure you instruct the user to configure variables locally
# os.environ["GEMINI_API_KEY"] = "YOUR_KEY"
# os.environ["SERPER_API_KEY"] = "YOUR_KEY"

print("--- DealScout AI Agent Terminal Interface ---")
print("Type 'exit' or 'quit' to stop the loop.\n")

config = {"configurable": {"thread_id": "cli_session"}}

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
        
    inputs = {"messages": [{"role": "user", "content": user_input}]}
    
    # Run the agent stream
    for output in app.stream(inputs, config, stream_mode="values"):
        final_msg = output["messages"][-1]
    
    print(f"Agent: {final_msg.content}\n")