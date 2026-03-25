from typing import TypedDict, Annotated
import operator
import os

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_groq import ChatGroq

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ── State ─────────────────────────────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# ── System Prompt ─────────────────────────────────────────────────────────────
system_prompt = SystemMessage(content=(
    "You are Felix, a helpful and friendly AI assistant. "
    "Answer clearly and concisely."
))

# ── Node ──────────────────────────────────────────────────────────────────────
def chatbot_node(state: ChatState):
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# ── Graph ─────────────────────────────────────────────────────────────────────
memory = MemorySaver()
graph = StateGraph(ChatState)
graph.add_node("chatbot", chatbot_node)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)
chatbot = graph.compile(checkpointer=memory)
