# My Agents Collection

This repository contains four distinct AI agent projects, each demonstrating different capabilities using modern LLM and retrieval technologies. Below you’ll find detailed descriptions, usage, and the technologies powering each project.

---

## 1. Drafter

**Description:**  
Drafter is an interactive document editing assistant. It helps users create, update, and save documents through a conversational interface. The agent can update the document content and save it to a file, leveraging LLMs for natural language understanding.

**Key Features:**
- Update document content interactively.
- Save the document to a text file.
- Shows the current document state after each modification.

**Technologies Used:**
- `langchain` (core, graph, tools)
- `langchain-google-genai` (Gemini 1.5 Flash model)
- `python-dotenv` (for environment variable management)

**How it works:**  
The agent uses a state graph to manage conversation flow, with tool nodes for updating and saving documents. It uses Google’s Gemini LLM for language understanding and tool invocation.

---

## 2. Memory Agent

**Description:**  
Memory Agent is a simple conversational agent that maintains a conversation history. It uses an LLM to generate responses and appends them to the ongoing message list, simulating memory.

**Key Features:**
- Maintains full conversation history.
- Generates context-aware responses.

**Technologies Used:**
- `langchain` (core, graph)
- `langchain-google-genai` (Gemini 1.5 Flash model)
- `python-dotenv`

**How it works:**  
The agent uses a state graph with a single processing node. Each user input is appended to the message history, and the LLM generates a response based on the entire conversation.

---

## 3. RAG Agent

**Description:**  
The RAG (Retrieval-Augmented Generation) Agent answers questions about the "Stock Market Performance 2024" PDF. It uses retrieval-augmented generation to find and cite relevant information from the document.

**Key Features:**
- Loads and splits a PDF into retrievable chunks.
- Embeds and stores document chunks in a Chroma vector database.
- Answers user questions by retrieving and citing relevant passages.

**Technologies Used:**
- `langchain` (core, graph, text splitter)
- `langchain-openai` (GPT-4o, OpenAI Embeddings)
- `langchain-chroma`, `chromadb`
- `langchain-community` (PDF loader)
- `pypdf`
- `python-dotenv`

**How it works:**  
The agent loads the PDF, splits it into chunks, embeds them, and stores them in a Chroma vector store. When a user asks a question, it retrieves the most relevant chunks and uses an LLM to generate a cited answer.

---

## 4. ReAct Agent

**Description:**  
The ReAct Agent demonstrates tool-augmented reasoning. It can perform arithmetic operations (add, subtract, multiply) and answer user queries, combining tool use with LLM reasoning.

**Key Features:**
- Supports addition, subtraction, and multiplication via tool calls.
- Handles multi-step reasoning and tool invocation.
- Provides conversational responses.

**Technologies Used:**
- `langchain` (core, graph, tools)
- `langchain-google-genai` (Gemini 1.5 Flash model)
- `python-dotenv`

**How it works:**  
The agent uses a state graph with tool nodes for arithmetic operations. The LLM decides when to invoke tools and when to respond directly, following the ReAct (Reason + Act) paradigm.

---

## Data and Other Files

- **Stock_Market_Performance_2024.pdf:** Used by the RAG Agent for retrieval-based Q&A.
- **chroma.sqlite3, chroma_db/, d8183236-1022-43e5-ab81-edc1cd016fe1/:** Vector database and data files for document retrieval.
- **requirements.txt:** Lists all Python dependencies for the agents.

---

## Installation

1. **Clone the repository:**
   ```
   git clone <repo-url>
   cd my-agents
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Create a `.env` file with your API keys (for Google Gemini and/or OpenAI).

---

## Usage

- **Drafter:**  
  ```
  python drafter.py
  ```
- **Memory Agent:**  
  ```
  python memory_agent.py
  ```
- **RAG Agent:**  
  ```
  python rag_agent.py
  ```
- **ReAct Agent:**  
  ```
  python ReAct.py
  ```

---

## Technologies Used

- [LangChain](https://github.com/langchain-ai/langchain) (core, graph, tools, community)
- [LangChain Google GenAI](https://github.com/langchain-ai/langchain-google-genai)
- [LangChain OpenAI](https://github.com/langchain-ai/langchain-openai)
- [ChromaDB](https://www.trychroma.com/)
- [PyPDF](https://pypdf.readthedocs.io/)
- [python-dotenv](https://github.com/theskumar/python-dotenv)

---

## License

This project is licensed under the MIT License. 