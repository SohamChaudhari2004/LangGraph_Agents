from typing import TypedDict , List
from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph ,START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    message: List[HumanMessage]

llm = ChatMistralAI(
    model_name = "mistral-small-latest",
    
    
)

def process(state:AgentState) -> AgentState:
    response = llm.invoke(state['message'])
    print(response)
    return state    

graph = StateGraph(AgentState)
graph.add_node("process", process) 
graph.add_edge(START , "process") 
graph.add_edge("process", END) 
agent = graph.compile()

user_ip = input("enter a prompt: ")
while user_ip != "exit":
    agent.invoke({"message":[HumanMessage(content=user_ip)]})
    user_ip = input("enter a prompt: ")

