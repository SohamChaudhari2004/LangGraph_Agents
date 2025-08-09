from typing import Annotated, Sequence , TypedDict
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int) -> int:
    """This function adds two numbers."""
    return a + b
@tool
def subtract(a: int, b: int) -> int:
    """This function subtracts two numbers."""
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """This function multiplies two numbers."""
    return a * b
tools = [add, subtract, multiply]

llm = ChatMistralAI(
    model_name='mistral-large-latest'   
).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    """This function handles the LLM response."""
    system_prompt = SystemMessage(
        content="You are a helpful assistant. Use the available tools for any calculations or actions. Do not answer directly if a tool can be used."
    )
    response = llm.invoke([system_prompt] + state['messages'])
    # Append the response to the message list
    return {'messages': state['messages'] + [response]}


def should_continue(state: AgentState) :
    """This function checks if the conversation should continue."""
    messages = state['messages']
    if not messages:
        return "end"
    last_message = messages[-1]
    # check if there is any need to call a tool if so continue, otherwise end
    if not getattr(last_message, "tool_calls", None):
        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools = tools, name="tool_node")
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,{
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "our_agent")
app = graph.compile()


# helper function to print the stream of messages
def print_stream(stream):
    for s in stream:
        message = s['messages'][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": [HumanMessage(content="add 2 and 4 and then multiply 3 to the result")]}
print_stream(app.stream(inputs, stream_mode="values"))