import os
import pandas as pd
from dotenv import load_dotenv
from typing import Annotated

from langchain.tools import StructuredTool
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from helpers import StudentRAGHelper
from langchain_groq import ChatGroq

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model = "llama3-8b-8192",)

class state(TypedDict):
    #messages have the format of a list of strings. the 'add_messages' function will
    #automatically add the messages to the state. in the annotation,defines how the 
    # messages are stored in the state.
    messages: Annotated[list[str], add_messages]
    # in this case, it appends the messages to the list, rather than replacing it.
    # the 'messages' key is a list of strings, which will be used to store the messages.
    
# Initialize Student RAG Helper
# -------------------------
rag_helper = StudentRAGHelper()

# -------------------------
# Define RAG tool
# -------------------------
def rag_query(query: str) -> str:
    """Retrieve student performance data based on query"""
    docs = rag_helper.retrieve(query, top_k=3)
    return "\n".join(docs)

rag_tool = StructuredTool.from_function(
    func=rag_query,
    name="student_progress_lookup",   
    description="Lookup student academic progress, marks, attendance, and assignments from the dataset."
)


tools = [rag_tool]
# List of tools to be used in the graph
llm_with_tools = llm.bind_tools(tools)
# Bind the tools to the LLM
# This will allow the LLM to use the tools in its responses

# Node function to handle the START state
def tool_calling_llm(state: state) -> str:
    """_summary_

    Args:
        state (state): _description_

    Returns:
        str: _description_
    """
    # this function will be called when the graph reaches the END state
    # it will return the messages stored in the state as a single string
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(state)
builder.add_node(
    "tool_calling_llm",
    tool_calling_llm 
)
builder.add_node(
    "tools",
    ToolNode(tools)
)

#add the edges to the graph
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
# if the tool calling LLM returns a tool call, it will go to the tools node
# if the LLM does not return a tool call, it will go to the END state
builder.add_edge("tools", END)

#compile the graph
graph = builder.compile()

def main():
    print("💬 Student RAG Chatbot (LangGraph + Groq)\nType 'exit' to quit.\n")
    print("Examples: \n"
          " - What is my overall progress in Data Structures?\n"
          " - How many assignments are pending for Ganesh?\n"
          " - Who are the top 3 students in AI?\n")

    while True:
        try:
            user = input("You: ").strip()
            if user.lower() in {"exit", "quit"}:
                print("Bot: Goodbye! 👋")
                break
            # LangChain agent with function calling; will call tools as needed
            response = graph.invoke({"messages": [user]})
            print(f"Bot: {response['messages'][-1].content}\n")
        except KeyboardInterrupt:
            print("\nBot: Goodbye! 👋")
            break
        except Exception as e:
            print(f"Bot (error): {e}\n")


if __name__ == "__main__":
    main()