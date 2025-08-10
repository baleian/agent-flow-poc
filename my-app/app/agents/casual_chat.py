import os
from typing import Literal
import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

system_message_prompt = """
당신은 'Cudori'라는 이름의 대화형 AI입니다.

사용자의 질문에 답변하기 위해 다음 단계를 따르세요:
1. 먼저, 질문에 답하기 위해 사용 가능한 도구가 필요한지 생각합니다.
2. **만약 도구가 필요하다면**, 필요한 도구를 호출하세요.
3. **만약 도구가 필요 없다면**, 당신의 내부 지식을 사용하여 사용자의 질문에 직접 답변하세요.
""".rstrip()


@tool
def get_current_time() -> str:
    """
    현재 시각을 가져올 때 사용합니다.
    예: '지금 몇시야?', '오늘 몇일이야?'
    """
    now = datetime.datetime.now()
    return now.isoformat()

@tool
def get_weather(city: Literal["seoul", "newyork"], current_time: str) -> str:
    """특정 도시의 현재 날씨 정보를 가져옵니다.
    Args:
    - city: 'seoul' 또는 'newyork'.
    - current_time: 현재 시각을 나타내는 '%Y-%m-%d %H:%M:%S' 포맷의 문자열.
    """
    if city.lower() == "seoul":
        return "맑음, 28°C"
    elif city.lower() == "newyork":
        return "안개, 15°C"
    else:
        return "알 수 없는 도시입니다."

# 테스트: 서울과 뉴욕의 현재 날씨를 비교하고, 지금 여행하기 더 좋은 곳을 추천해줘.

tools = [get_current_time, get_weather]
model = ChatOllama(
    model=os.environ["MODEL_NAME"], 
    base_url=os.environ["MODEL_BASE_URL"], 
    reasoning=True
)
model = model.bind_tools(tools)

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(system_message_prompt),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain = prompt_template | model


def casual_chat(state: MessagesState):
    response = chain.invoke(state)
    if "reasoning_content" in response.additional_kwargs:
        del response.additional_kwargs["reasoning_content"]
    return {"messages": [response]}

def tools_condition(state: MessagesState) -> Literal["Casual_Chat.tools", "__end__"]:
    ai_message = state['messages'][-1]
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "Casual_Chat.tools"
    return END


workflow = StateGraph(state_schema=MessagesState)

workflow.add_node("Casual_Chat", casual_chat)
workflow.add_node("Casual_Chat.tools", ToolNode(tools))

workflow.add_edge(START, "Casual_Chat")
workflow.add_conditional_edges("Casual_Chat", tools_condition)
workflow.add_edge("Casual_Chat.tools", "Casual_Chat")
workflow.add_edge("Casual_Chat", END)

graph = workflow.compile()
