import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_mistralai import ChatMistralAI , MistralAIEmbeddings
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_community.vectorstores import chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



load_dotenv()


llm = ChatMistralAI(model="mistral-large-latest" , temperature=0)

embeddings = MistralAIEmbeddings(model="mistral-embed")

pdf_path = "AiScaffolder.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"The file {pdf_path} does not exist.")

# load the PDF document
pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"Loaded {len(pages)} pages from the PDF.")
except Exception as e:
    raise RuntimeError(f"Failed to load PDF: {str(e)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

pages_split = text_splitter.split_documents(pages)

persist_dir = "chroma_db"
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

collection_name = "AiScaffolder"


vectorstore = chroma.Chroma.from_documents(
    documents=pages_split,
    collection_name=collection_name,
    embedding=embeddings,
    persist_directory=persist_dir
)




retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.5}  # You can adjust the threshold as needed
)

@tool
def retriever_tool(query: str) -> str:
    """ this tool retrieves relevant documents based on the query."""
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant documents found."
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    return "\n\n".join(results)
tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state:AgentState) :
    """This function checks if the conversation should continue."""
    result = state['messages'][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the provided documents.
Use the retriever tool available to answer questions about the Ai scaffolding project data. You can also ask follow-up questions based on the retrieved information.
If you need to look up some information before asking a follow up question, you are allowed to do so.
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict  = {tool.name: tool for tool in tools}

def call_llm(state: AgentState) :
    """This function handles the LLM response."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(messages)
    
    # Append the response to the message list
    return {'messages': state['messages'] + [response]}

# retriever agent
def take_action(state: AgentState) -> AgentState:
    """execute tool calls from llm response."""
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Executing tool: {t['name']} with args: {t['args'].get('query', 'no query provided')}")
        if t['name'] not in tools_dict:
            raise ValueError(f"Tool {t['name']} not found in tools dictionary.")
        tool_result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
        print(f" results length returned: {len(tool_result)}")
        results.append(
            ToolMessage(
                tool_call_id=t['id'],
                content=str(tool_result),
                name=t['name'],
                args=t['args']
            )
        )
    print('tools executed successfully')
    # Append tool messages to the existing messages
    return {'messages': state['messages'] + results}


graph = StateGraph(AgentState)
graph.add_node("call_llm", call_llm)
graph.add_node("take_action", take_action)
graph.add_conditional_edges(
    "call_llm",
    should_continue,
    {True: "take_action", False: END}
)
graph.add_edge("take_action", "call_llm")
graph.set_entry_point("call_llm")

app = graph.compile()


def running_agent():
    print("\n ===== RAG AGENT =====")
    print("This agent uses a retriever to answer questions based on the provided documents.")
    print("You can ask questions about the stock market performance in 2024.")
    print("Type 'exit' to end the conversation.\n")
    
    state = {"messages": []}
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        message = HumanMessage(content=user_input)

        results = app.invoke({'messages': message})

        for message in state["messages"]:
           if isinstance(message, AIMessage):
               print(f"\n🤖 AI: {message.content}")
           elif isinstance(message, ToolMessage):
                print(f"🔧 TOOL RESULT: {message.content}")
        
        print("\n--- End of Response ---\n")
        print(results['messages'][-1].content)


if __name__ == "__main__":
    running_agent()