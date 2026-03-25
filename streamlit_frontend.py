import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langgraph_backend import chatbot
import uuid

st.set_page_config(page_title="Chat With Felix", page_icon="🤖", layout="centered")
st.title("🤖 Chat With Felix")
st.write("A chatbot built with GenAI and LangGraph. Ask me anything!")

# ── Session State ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

# ── Chat Function ──────────────────────────────────────────────────────────────
def chat_with_bot(user_input):
    st.session_state["messages"].append(HumanMessage(content=user_input))

    response = chatbot.invoke(
        {"messages": st.session_state["messages"]},
        config={"configurable": {"thread_id": st.session_state["thread_id"]}}
    )

    ai_reply = response["messages"][-1].content
    st.session_state["messages"].append(AIMessage(content=ai_reply))

# ── UI ─────────────────────────────────────────────────────────────────────────
user_text = st.chat_input("💬 Type your message here...")

if user_text:
    chat_with_bot(user_text)
    st.rerun()

# ── Chat History ───────────────────────────────────────────────────────────────
st.subheader("Chat History")
if not st.session_state["messages"]:
    st.info("No messages yet. Start chatting!")
else:
    for msg in st.session_state["messages"]:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)