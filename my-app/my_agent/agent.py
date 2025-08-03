from langgraph.graph import START, END, StateGraph

from my_agent.utils.state import AgentState
from my_agent.utils.nodes import chatbot, tool_node, tools_condition


workflow = StateGraph(state_schema=AgentState)
workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges("chatbot", tools_condition)
workflow.add_edge("tools", "chatbot")

graph = workflow.compile()
