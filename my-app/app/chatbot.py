from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage

from app.agents import supervisor


def chatbot(state: MessagesState):
    response = supervisor.graph.invoke(state)
    ai_message = response["messages"][-1]
    if isinstance(ai_message, AIMessage):
        # Append only the last AI message to state (to minimize context)
        return {"messages": ai_message}

def make_chatbot_graph():
    workflow = StateGraph(MessagesState)
    workflow.add_node("Cudori", chatbot)
    workflow.add_edge(START, "Cudori")
    workflow.add_edge("Cudori", END)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
