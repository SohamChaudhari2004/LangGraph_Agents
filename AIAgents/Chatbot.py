from typing import TypedDict , List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph ,START, END
from dotenv import load_dotenv
from langchain.agents import initialize_agent
# from langchain.memory import ConversationSummaryBufferMemory

load_dotenv()

class AgentState(TypedDict):
    message: List[Union[HumanMessage,AIMessage]]


llm  = ChatMistralAI(
    model_name= 'mistral-small-latest'
)


def process(state:AgentState)-> AgentState:
    """This functions handles llm reponse"""
    response = llm.invoke(state['message'])
    state['message'].append(AIMessage(content=response.content))
    print("AI message: ",response.content)
    print("current state :" , state["message"])

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process) 
graph.add_edge(START , "process") 
graph.add_edge("process", END) 

agent = graph.compile()

conversation_history = []

user_ip = input("enter anything : \n")
while user_ip != "exit":
    conversation_history.append(HumanMessage(content=user_ip))

    result = agent.invoke({"message" : conversation_history})
    print(f"\n {result['message']} \n")
    conversation_history = result["message"]
    user_ip = input("enter anything : \n")
    

with open('logging.txt','w') as file:
    file.write("your converstation log")
    for message in conversation_history:
        if isinstance(message , HumanMessage):
            file.write(f"YOU: {message.content}")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}")

print("Conversation logged in logging.txt")

# or can use 
# from langchain.memory import ConversationSummaryBufferMemory
# memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000)
# agent_with_memory = initialize_agent(
#     llm=llm,
#     agent_type="chat-conversational-react-description",
#     memory=memory,
#     verbose=True
# )