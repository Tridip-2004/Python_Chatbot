# 🤖 Voice-Enabled Chatbot using LangGraph & Streamlit
📌 Project Overview

This project is a simple conversational chatbot built using LangGraph for structured conversation flow and Streamlit for an interactive web interface.
The chatbot supports text-based interaction and can speak responses using Text-to-Speech (TTS), making it more engaging and accessible.

# 🎯 Features

💬 Interactive chatbot UI using Streamlit

🧠 Conversation flow management using LangGraph

🔊 Voice input(speech-to-text) for chatbot responses

🔄 Maintains conversation state

⚡ Lightweight and easy to run locally

# 🛠️ Tech Stack

Programming Language: Python

Frameworks & Libraries:

Streamlit

LangGraph

LangChain

Groq API (or compatible LLM)

pyttsx3 (for Text-to-Speech)

# 🧩 Architecture Overview
User Input (Text)
        ↓
Streamlit UI
        ↓
LangGraph (Conversation Flow)
        ↓
LLM Response
        ↓
Text Output + Voice Output (TTS)
