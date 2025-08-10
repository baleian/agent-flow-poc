import os
import json
from typing import Literal, List, Dict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph, START, END

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, ToolCall, trim_messages
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode

from app.tools.dw import get_table_schemas, execute_query


text_to_sql_tools = [get_table_schemas]
sql_corrector_tools = [get_table_schemas, execute_query]
sql_executor_tools = [execute_query]


def make_text_to_sql_chain():
    system_prompt = """
당신은 전문가 수준의 Text-to-SQL 에이전트입니다. 당신의 유일한 임무는 사용자의 자연어 질문을 정확하고 실행 가능한 SQL 쿼리로 변환하는 것입니다.

**당신의 작업 절차**:
1. **스키마 발견 (가장 중요한 첫 단계)**: 사용자의 질문을 분석한 후, 반드시 `get_table_schemas` 도구를 먼저 호출하여 질문과 관련된 테이블 스키마 정보를 얻어야 합니다. 절대 테이블이나 컬럼 이름을 스스로 추측해서는 안 됩니다.
2. **스키마 기반 SQL 생성**: `get_table_schemas` 도구를 통해 얻은 테이블 이름, 컬럼명, 데이터 타입 등 실제 스키마 정보를 바탕으로 SQL 쿼리를 작성합니다.
3. **정확한 쿼리 작성**: 사용자의 의도를 정확히 반영하는 SQL을 생성하세요. 올바른 JOIN, WHERE 절, 집계 함수를 사용해야 합니다.
  - 사용자의 질문에 문자열 기반의 조건절(`WHERE`)이 포함되어 있습니까?
  - 그렇다면, 사용자가 질문에서 언급한 값을 `WHERE` 절에 직접 사용해야 합니다. 어떤 값이 있는지 **스스로 추측하지 마세요**.
4. **SQLite 문법 준수**: 생성된 모든 SQL 쿼리는 SQLite 데이터베이스에서 실행 가능해야 합니다. 표준 SQL을 우선적으로 사용하되, 날짜/시간 함수(`strftime` 등)와 같이 SQLite의 특정 구문을 따라야 할 경우 이를 반드시 준수하세요.

**출력 형식**:
당신의 최종 응답은 오직 **순수한 SQL 쿼리 문자열**이어야 합니다. 어떠한 설명, 인사, 주석, 마크다운 코드 블록(```sql)도 포함하지 마세요.
    """.rstrip()
        
#     system_prompt = """
# 당신은 **탐정(Detective)**과 같은 능력을 지닌 전문가 수준의 Text-to-SQL 에이전트입니다. 당신의 유일한 임무는 사용자의 자연어 질문을 실제 데이터를 기반으로 검증하여 가장 정확하고 실행 가능한 SQL 쿼리로 변환하는 것입니다.

# **당신의 작업 절차 (계층적 추론)**:
# 1. **스키마 발견 (현장 파악)**: 사용자의 질문을 분석한 후, 반드시 `get_table_schemas` 도구를 먼저 호출하여 질문과 관련된 테이블 스키마 정보를 얻어야 합니다. 이는 수사의 기본이며, 절대 건너뛰어서는 안 됩니다.
# 2. **데이터 탐색 (증거 수집)**:
#   - 사용자의 질문에 상태, 유형, 분류, 지역명, 일반 명사 등의 카테고리형 컬럼 기반의 조건절(`WHERE`)이 포함되어 있습니까? 그렇다면, 다음과 같이 탐색 전략을 따라 실제 데이터베이스의 값을 참조하도록 합니다.
#     1) 먼저 조건이 되는 컬럼의 테이블에서 예상되는 값으로 조회를 시도합니다.
#         - 예시 쿼리: `SELECT COUNT(*) FROM orders WHERE status = "DONE";`
#     2) 만약 일치하는 결과가 없다면, 탐색 쿼리를 통해 실제 값의 종류를 파악합니다.
#         - 예시 쿼리: `SELECT status, COUNT(*) FROM orders ORDER BY COUNT(*) DESC LIMIT 200;` (상태 컬럼의 종류 파악)
#     3) 이 탐색 결과를 바탕으로, 사용자의 질문(예: "완료된 주문")이 실제 데이터베이스의 어떤 값 (예: `completed`, `finished`, `DONE` 등)에 해당하는지 정확히 판단하세요.
#     4) 만약 해당하는 값을 찾을 수 없는 경우, 사용자가 질문에서 언급한 값을 `WHERE` 절에 직접 사용해야 합니다.
# 3. **최종 쿼리 생성 (결론 도출)**: 1단계(스키마)와 2단계(데이터 증거)에서 얻은 모든 정보를 종합하여, 사용자의 의도를 완벽하게 반영하는 최종 SQL 쿼리 하나를 생성합니다. 올바른 JOIN, WHERE 절, 집계 함수를 사용하세요.
# 4. **SQLite 문법 준수**: 생성된 모든 SQL 쿼리는 SQLite 데이터베이스에서 실행 가능해야 합니다. 표준 SQL을 우선적으로 사용하되, 날짜/시간 함수(`strftime` 등)와 같이 SQLite의 특정 구문을 따라야 할 경우 이를 반드시 준수하세요.

# **출력 형식**:
# 당신의 최종 응답은 오직 **단 하나의 실행 가능한 SQL 쿼리 문자열**이어야 합니다.
#     """.rstrip()

    model = ChatOllama(
        model=os.environ["MODEL_NAME"], 
        base_url=os.environ["MODEL_BASE_URL"], 
        reasoning=True
    ).bind_tools(text_to_sql_tools)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    
    chain = prompt_template | model
    return chain


def make_sql_corrector_chain():
    system_prompt = """
당신은 **SQL 교정 전문가(SQL Corrector)**입니다. 당신의 유일한 임무는 입력으로 주어진 가상의 SQL 쿼리문에서 **카테고리형 컬럼에 대한 `WHERE` 조건의 문자열 값이 실제 데이터베이스에 존재하는 값과 일치하도록 검증하고 수정**하는 것입니다.

**가장 중요한 원칙(!IMPORTANT!): 불신과 검증 (Core Principle: Distrust and Verify)**
초기 쿼리의 `WHERE` 조건에 있는 값은 **항상 부정확하다고 가정하십시오.** 당신의 임무는 이 잠재적으로 틀린 값을 그대로 사용하는 것이 아니라, 이어지는 절차에 따라 **철저한 검증을 통해 올바른 실제 값으로 반드시 교정**하는 것입니다.
---
## 작업 절차 및 규약
**1. 1차 조사: 스키마 확인 (Schema Investigation)**
가장 먼저 `get_table_schemas` 도구를 호출하여 쿼리에 명시된 테이블들의 스키마 정보를 확보해야 합니다.
- **도구 사용:** `get_table_schemas(query="{입력 쿼리 원문}")`
- **핵심 확인 사항:** `WHERE` 절에 사용된 컬럼의 `description` 필드에 값의 종류가 명시적으로 열거되어 있는지 확인합니다.
- **판단:**
  - 만약 `description` 정보만으로 실제 값을 명확히 알 수 있다면, 그 정보를 바탕으로 즉시 쿼리를 수정합니다. 이는 가장 효율적인 해결 경로입니다.
  - `description에` 정보가 없거나 불충분할 경우, 2단계인 '데이터 직접 탐색'으로 넘어갑니다.

**2. 2차 조사: 데이터 직접 탐색 (Live Data Exploration)**
스키마 정보만으로 값의 유효성을 판단할 수 없을 때, `execute_query` 도구를 사용하여 데이터베이스를 직접 탐색합니다. **절대 추측에 의존해서는 안 됩니다.**
- **A. 가설 검증 (Hypothesis Testing):** 먼저, 입력 쿼리에 있는 값이 실제로 존재하는지 최소한의 비용으로 확인합니다.
  - **예시:** 초기 쿼리가 `SELECT ... WHERE status = 'DONE'`일 경우, 다음과 같은 `COUNT` 쿼리를 실행하여 'DONE'의 존재 유무를 빠르게 확인합니다.
    - **탐색 쿼리:** `SELECT COUNT(*) FROM orders WHERE status = 'DONE'`
  - **판단:** 만약 이 쿼리의 결과가 0이라면, 해당 값은 존재하지 않을 가능성이 높으므로 다음 단계로 넘어갑니다.
- **B. 심층 탐사 (Deep Dive):** 가설 검증에 실패하면, 해당 컬럼이 가질 수 있는 실제 값들의 목록과 분포를 확인하여 사용자의 의도와 가장 일치하는 값을 찾아냅니다.
  - **예시:** `status` 컬럼의 실제 값을 알아내기 위해 다음과 같은 그룹화 쿼리를 실행합니다.
    - **탐색 쿼리:** `SELECT status, COUNT(*) FROM orders GROUP BY status ORDER BY COUNT(*) DESC LIMIT 100`
  - **판단:** 탐색 결과를 바탕으로 사용자가 의도한 'DONE'(완료)과 가장 유사하거나 가능성이 높은 실제 값(예: 'completed', 'finished', 'done' 등)을 찾아냅니다.
    - 예를 들면 'DONE'과 'done'은 의미적으로 완전히 같지만, Exact Match 할 경우 Case Sensitive로 인해 매칭에 실패합니다. 적절히 의도를 파악하여 쿼리를 수정하세요.
    - 조건 컬럼 선택에 **모호성이 존재하여 추가 검증이 필요한 경우 데이터 탐색을 반복**할 수 있습니다.

**3. 최종 쿼리 생성 (Final Query Generation)**
1, 2 단계에서 수집한 모든 증거(스키마 정보, 데이터 탐색 결과)를 종합하여, `WHERE` 조건절이 실제 데이터베이스 값에 기반하도록 수정된 최종 SQL 쿼리 하나를 생성합니다.
- **복잡한 쿼리 예시:**
  - **초기 쿼리:** `SELECT COUNT(T2.account_id) FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id WHERE T1.A3 = 'East Bohemia' AND T2.frequency = 'POPLATEK PO OBRATU'`
  - 수행 작업: 위 작업 절차에 따라 district 테이블의 A3 컬럼과 account 테이블의 frequency 컬럼 값의 유효성을 각각 확인하고, 필요시 모두 수정해야 합니다.
---
## 최종 출력 형식
- 당신의 최종 응답은 오직 **초기 쿼리에서 수정된의 실제 실행 가능한 SQL 쿼리 문자열**이어야 합니다.
"""
    model = ChatOllama(
        model=os.environ["MODEL_NAME"], 
        base_url=os.environ["MODEL_BASE_URL"], 
        reasoning=True
    ).bind_tools(sql_corrector_tools)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "초기 쿼리: {generated_sql}")
        ]
    )
    
    chain = prompt_template | model
    return chain


def make_sql_executor_chain():
    system_prompt = """
당신은 'Secure Query Executor'라는 이름의 데이터베이스 인터페이스 에이전트입니다. 당신의 최우선 순위는 **데이터베이스의 안정성을 보장**하는 것이며, 오직 안전하다고 검증된 **읽기 전용(read-only)** 쿼리만을 실행합니다.

**쿼리 처리 프로토콜 (반드시 순서대로 따를 것)**:
1. **위험 평가 (Risk Assessment)**:
  - 입력된 쿼리가 읽기 전용 (`SELECT`) 쿼리 인지 확인합니다.
  - `UPDATE`, `DELETE`, `INSERT`, `CREATE`, `DROP`, `ALTER`, `TRUNCATE` 등 데이터 수정/삭제 관련 키워드가 포함되어 있는지 확인합니다.
  - 만약 하나라도 금지된 키워드가 발견되면, 다른 모든 단계를 즉시 건너뛰고 **"데이터 조회를 위한 SELECT 쿼리만 실행 가능합니다."** 라는 메시지를 반환하세요.

2. **유효성 검증 (Validation)**:
  - 보안 검사를 통과한 쿼리가 SQLite 문법 표준을 준수하는지 확인합니다. 문법 오류가 예상되면, **"입력된 SQL의 문법이 올바르지 않습니다."** 라고 응답하세요.

3. **실행 위임 (Execution Delegation)**:
  - 모든 검증을 통과한 쿼리는 `execute_query` **도구를 사용**하여 데이터베이스로 전달합니다. 당신이 직접 쿼리를 실행하는 것이 아니라, 도구에 위임하는 역할입니다.
    """.rstrip()

    model = ChatOllama(
        model=os.environ["MODEL_NAME"], 
        base_url=os.environ["MODEL_BASE_URL"], 
        reasoning=True
    ).bind_tools(sql_executor_tools)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(system_prompt),
            ("human", "{generated_sql}")
        ]
    )
    
    chain = prompt_template | model
    return chain

def make_summary_chain():
    system_prompt = """
- 당신은 `Query Result`의 데이터를 분석합니다. 
- 사용자가 이해하기 쉬운 **핵심적인 내용을 요약**하여, 간결하고 명확한 문장으로 최종 보고서를 작성합니다.
- 사용자의 질문에 대한 대답을 할 수 있는 요약을 작성합니다. 예를 들어, "총 5개의 제품이 검색되었으며, 가장 인기 있는 제품은 'A'입니다." 와 같이 요약해 주세요.
- 만약 사용자 질문에 정확히 대답할 수 없다면, 정확한 정보를 확인할 수 없다고 알려주세요.
    """.rstrip()

    model = ChatOllama(
        model=os.environ["MODEL_NAME"], 
        base_url=os.environ["MODEL_BASE_URL"], 
        reasoning=False
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("human", """
`User Question`: {user_question}

`Generated SQL`: {generated_sql}

`Query Result`: {query_result}
             """.rstrip())
        ]
    )
    
    chain = prompt_template | model
    return chain


text_to_sql_chain = make_text_to_sql_chain()
sql_corrector_chain = make_sql_corrector_chain()
sql_executor_chain = make_sql_executor_chain()
summary_chain = make_summary_chain()


class TextToSQLState(MessagesState):
    user_question: str
    generated_sql: str
    query_result: List[Dict]


def text_to_sql(state: TextToSQLState):
    response = text_to_sql_chain.invoke(state)
    if "reasoning_content" in response.additional_kwargs:
        del response.additional_kwargs["reasoning_content"]
    update_state = {"messages": [response]}
    if isinstance(state["messages"][-1], HumanMessage):
        update_state["user_question"] = state["messages"][-1].content
    if response.content:
        update_state["generated_sql"] = response.content
    return update_state


def sql_corrector(state: TextToSQLState):
    if isinstance(state["messages"][-1], HumanMessage):
        messages = [state["messages"][-1]]
    else:
        messages = state["messages"]
    response = sql_corrector_chain.invoke({"messages": messages, "generated_sql": state["generated_sql"]})
    if "reasoning_content" in response.additional_kwargs:
        del response.additional_kwargs["reasoning_content"]
    update_state = {"messages": [response]}
    if response.content:
        update_state["generated_sql"] = response.content
    return update_state


def sql_executor(state: TextToSQLState):
    if isinstance(state["messages"][-1], ToolMessage):
        query_result = json.loads(state["messages"][-1].content)
        if isinstance(query_result, dict) and "data" in query_result:
            return {"query_result": query_result.get("data", [])}
        
    response = sql_executor_chain.invoke(state)
    if "reasoning_content" in response.additional_kwargs:
        del response.additional_kwargs["reasoning_content"]
    return {"messages": [response]}


def summary(state: TextToSQLState):
    response = summary_chain.invoke(state)
    return {"messages": [response]}


def text_to_sql_tools_condition(state: TextToSQLState) -> Literal["Text_to_SQL.tools", "SQL_Corrector", "__end__"]:
    response = state["messages"][-1]
    if hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
        return "Text_to_SQL.tools"
    elif "generated_sql" in state:
        return "SQL_Corrector"
    else:
        return END
    
def sql_corrector_tools_condition(state: TextToSQLState) -> Literal["SQL_Corrector.tools", "SQL_Executor"]:
    response = state["messages"][-1]
    if hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
        return "SQL_Corrector.tools"
    else:
        return "SQL_Executor"
    
def sql_executor_tools_condition(state: TextToSQLState) -> Literal["SQL_Executor.tools", "Summary", "__end__"]:
    response = state["messages"][-1]
    if hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
        return "SQL_Executor.tools"
    elif "query_result" in state:
        return "Summary"
    else:
        return END


workflow = StateGraph(state_schema=TextToSQLState)

workflow.add_node("Text_to_SQL", text_to_sql)
workflow.add_node("Text_to_SQL.tools", ToolNode(text_to_sql_tools))
workflow.add_node("SQL_Corrector", sql_corrector)
workflow.add_node("SQL_Corrector.tools", ToolNode(sql_corrector_tools))
workflow.add_node("SQL_Executor", sql_executor)
workflow.add_node("SQL_Executor.tools", ToolNode(sql_executor_tools))
workflow.add_node("Summary", summary)

workflow.add_edge(START, "Text_to_SQL")
workflow.add_conditional_edges("Text_to_SQL", text_to_sql_tools_condition)
workflow.add_edge("Text_to_SQL.tools", "Text_to_SQL")
workflow.add_conditional_edges("SQL_Corrector", sql_corrector_tools_condition)
workflow.add_edge("SQL_Corrector.tools", "SQL_Corrector")
workflow.add_conditional_edges("SQL_Executor", sql_executor_tools_condition)
workflow.add_edge("SQL_Executor.tools", "SQL_Executor")
workflow.add_edge("Summary", END)

graph = workflow.compile()
