from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools([])


embeddings = OpenAIEmbeddings()

pdf_path = "Stock_Market_Performance_2024.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError("PDF file not found.")

loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"PDF has been loaded and has {len(docs)} pages")


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

persist_directory = "chroma_db"
collection_name = "stock_market"
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name=collection_name
)
print("Created ChromaDB vector store!")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

@tool
def retriever_tool(query: str) -> str:
    """
    Searches and returns information from the Stock Market Performance 2024 PDF.
    """
    results = retriever.invoke(query)
    if not results:
        return "No relevant information found."
    return "\n\n".join([doc.page_content for doc in results])

tools = [retriever_tool]
llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

system_prompt = """
You are an AI assistant specialized in answering questions about the Stock Market Performance 2024 document.
Use the retriever tool to find and reference data from the document.
Always cite relevant passages in your answer.
"""

def call_llm(state: AgentState) -> AgentState:
    msgs = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(msgs)
    return {"messages": [response]}

def should_continue(state: AgentState) -> bool:
    last = state["messages"][-1]
    return hasattr(last, "tool_calls") and last.tool_calls is not None and len(last.tool_calls) > 0

def take_action(state: AgentState) -> AgentState:
    tool_calls = state["messages"][-1].tool_calls
    outputs = []
    for call in tool_calls:
        tool_name = call["name"]
        args = call.get("args", {})
        tool_fn = next((t for t in tools if t.name == tool_name), None)
        if tool_fn:
            result = tool_fn.invoke(args.get("query", ""))
            outputs.append(ToolMessage(tool_call_id=call["id"], name=tool_name, content=result))
    return {"messages": outputs}

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever", take_action)
graph.add_conditional_edges("llm", should_continue, {True: "retriever", False: END})
graph.add_edge("retriever", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

def run_agent():
    print("=== RAG Agent Ready ===")
    while True:
        query = input("\nYour question: ")
        if query.lower() in ["exit", "quit"]:
            break
        messages = [HumanMessage(content=query)]
        response = rag_agent.invoke({"messages": messages})
        print("\n=== Answer ===")
        print(response["messages"][-1].content)

run_agent()
