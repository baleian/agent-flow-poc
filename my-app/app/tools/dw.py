import os
import requests

from langchain_core.tools import tool
from langchain_core.documents import Document
import grpc

from app.proto import document_search_pb2
from app.proto import document_search_pb2_grpc


@tool(response_format="content_and_artifact")
def get_table_schemas(query: str):
    """
    유저가 원하는 데이터를 조회하기 위해 필요한 연관성이 높은 테이블 스키마를 검색할 때 사용합니다.
    Join과 같은 복잡한 SQL이 요구되는 경우, 관련 테이블이 여러개 있을 수 있습니다.
    Parameters:
    - query: VectorStore에서 검색하기 위한 쿼리. 자연어 기반으로 검색할 수 있으므로, 핵심 사용자 질문에 해당하는 자연어을 그대로 사용하세요.
    """
    with grpc.insecure_channel(os.environ["DW_SEARCH_GRPC_CHANNEL"]) as channel:
        stub = document_search_pb2_grpc.DocumentSearchServiceStub(channel)
        responses = stub.RetrieveDocuments(document_search_pb2.DocumentSearchRequest(query=query, k=10))
        documents = [Document(page_content=res.payload['content'], id=res.payload['id'], metadata=res.payload['metadata']) for res in responses]
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in documents
        )
        return serialized, documents


@tool
def execute_query(sql: str) -> dict:
    """
    SQLite 데이터베이스에 SQL을 실행하고 쿼리 결과를 응답합니다.
    Parameters:
    - sql: SQLite에서 실행 가능한 SQL 문자열
    """
    url = os.environ['SQLITE_SERVER_URL'] + "/query"
    headers = {"Content-Type": "application/json"}
    payload = {"query": sql}
    response = requests.post(url, headers=headers, json=payload)
    return response.json()
