import grpc
from concurrent import futures
import time

from app.proto import document_search_pb2
from app.proto import document_search_pb2_grpc
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging
logging.basicConfig(level=logging.INFO)

EMBEDDING_MODEL = "jhgan/ko-sbert-nli"


class DocumentSearchService(document_search_pb2_grpc.DocumentSearchServiceServicer):
    def __init__(self):
        self.model_name = EMBEDDING_MODEL
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        logging.info("embeddings loaded.")
        self.vectorstore = FAISS.load_local("vectorstore", self.embeddings, allow_dangerous_deserialization=True)
        logging.info("vectorstore loaded.")

    def RetrieveDocuments(self, request, context):
        query = request.query
        k = request.k or 3
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        for document in retriever.invoke(query):
            response = document_search_pb2.DocumentSearchResponse()
            response.payload.update(dict(
                id=document.id,
                content=document.page_content,
                metadata=document.metadata
            ))
            yield response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    document_search_pb2_grpc.add_DocumentSearchServiceServicer_to_server(DocumentSearchService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logging.info("server started.")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
