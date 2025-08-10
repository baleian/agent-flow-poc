import os

from langchain_core.tools import tool
from langchain_core.documents import Document
import grpc

from app.proto import document_search_pb2
from app.proto import document_search_pb2_grpc


@tool(response_format="content_and_artifact")
def get_documents(query: str, count: int = 3):
    """
    유저의 질의에 가장 연관성이 높은 문서를 검색할 때 사용합니다.
    일반적인 질문이 아닌 사내 문서 데이터베이스에서 조회가 필요할 때 연관 문서를 가져올 수 있습니다.
    Parameters:
    - query: VectorStore에서 검색하기 위한 쿼리.
    - count: 연관 문서 상위 몇개를 가져올 지. 기본값: 3
    """
    with grpc.insecure_channel(os.environ["DOCUMENT_SEARCH_GRPC_CHANNEL"]) as channel:
        stub = document_search_pb2_grpc.DocumentSearchServiceStub(channel)
        responses = stub.RetrieveDocuments(document_search_pb2.DocumentSearchRequest(query=query, k=count))
        documents = [Document(page_content=res.payload['content'], id=res.payload['id'], metadata=res.payload['metadata']) for res in responses]
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in documents
        )
        return serialized, documents
