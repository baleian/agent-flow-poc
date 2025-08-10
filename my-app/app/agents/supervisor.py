import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import Tool
from pydantic import BaseModel, Field

from app.agents import document_qa, coder, text_to_sql, casual_chat

# import langchain
# langchain.debug = True


system_prompt = """
당신은 AI 에이전트 시스템의 지능형 라우팅 감독관입니다.

당신의 **유일한 임무**는 사용자의 최신 질문을 **내용이 아닌 유형에 따라** 분류하고, 가장 적합한 전문가에게 즉시 작업을 전달하는 것입니다.

**행동 원칙**
- **신속한 분류**: 질문의 내용을 깊게 이해하거나 해결 방법을 생각하지 마세요. 오직 어느 카테고리에 속하는지만을 신속하게 판단합니다.
- **답변 금지**: 어떠한 경우에도 사용자의 질문에 직접 답변하지 않습니다.
- **단일 책임**: 당신의 책임은 오직 '라우팅'입니다.

**전문가 목록**
- `Document_QA`: 사내 문서, 정책, 매뉴얼 등 특정 사내 정보에 대한 질문.
- `Coder`: 코드 작성, 수정, 분석, 디버깅과 관련된 질문.
- 'Text_to_SQL': 데이터 분석(EDA), 데이터베이스 조회, 통계 관련 질문일 경우
- `Casual_Chat`: 위 경우에 해당하지 않는 일반적인 대화, 인사, 잡담.

주어진 전문가 목록 중 하나를 반드시 선택하여 작업을 전달하세요.
""".rstrip()

class Route(BaseModel):
    """어떤 에이전트를 다음으로 호출할지 결정합니다."""
    next: str = Field(..., description="다음 경로. 옵션: [Document_QA, Coder, Casual_Chat]")

llm = ChatOllama(
    model=os.environ["MODEL_NAME"], 
    base_url=os.environ["MODEL_BASE_URL"], 
    reasoning=True
)
llm = llm.bind_tools(tools=[Route], tool_choice="Route")
llm = llm.with_config(tags=["WANT_TO_STREAM"])

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain = prompt_template | llm


class AgentState(MessagesState):
    next: str

def root(state: AgentState):
    response = chain.invoke(state)
    route = response.tool_calls[0]['args']
    return {"next": route['next']}


workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", root)
workflow.add_node("Document_QA", document_qa.graph)
workflow.add_node("Coder", coder.graph)
workflow.add_node("Text_to_SQL", text_to_sql.graph)
workflow.add_node("Casual_Chat", casual_chat.graph)

workflow.add_edge(START, "Supervisor")
workflow.add_conditional_edges(
    "Supervisor",
    lambda state: state["next"],
    {
        "Document_QA": "Document_QA",
        "Coder": "Coder",
        "Text_to_SQL": "Text_to_SQL",
        "Casual_Chat": "Casual_Chat"
    }
)

graph = workflow.compile()

# memory = MemorySaver()
# graph = workflow.compile(checkpointer=memory)
