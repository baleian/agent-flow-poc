import shutil
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import logging
logging.basicConfig(level=logging.INFO)

EMBEDDING_MODEL = "jhgan/ko-sbert-nli"
VECTORSTORE_DIR = "vectorstore"
SCHEMA_FILES_DIR = "dataset/financial_db_schemas"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
logging.info("embeddings loaded.")

vectorstore_dir = Path(VECTORSTORE_DIR)

TABLES = [
    ("account", "Table containing customer account information."),
    ("card", "Table containing credit card information."),
    ("client", "Table containing client demographic information."),
    ("disp", "Table linking clients to accounts with specific rights (dispositions)."),
    ("district", "Table containing demographic and economic statistics for each district."),
    ("loan", "Table containing loan information for each account."),
    ("order", "Table containing payment order information."),
    ("trans", "Table containing detailed transaction records for each account."),
]

def load_document(source, **kwargs):
    with open(source, "r") as f:
        page_content = f.read()
        return Document(page_content=page_content, metadata=dict(source=source, **kwargs))

def load_documents():
    for table_name, table_description in TABLES:
        source = f"{SCHEMA_FILES_DIR}/{table_name}.sql"
        yield load_document(source=source, table_description=table_description)

def reload_vectorstore():
    if not vectorstore_dir.is_dir():
        documents = list(load_documents())
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(vectorstore_dir)
        logging.info("vectorstore saved.")
    vectorstore = FAISS.load_local(vectorstore_dir, embeddings, allow_dangerous_deserialization=True)
    logging.info("vectorstore loaded.")
    return vectorstore

vectorstore = reload_vectorstore()

def retrieve_documents(query, k=3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    for document in retriever.invoke(query):
        yield document



import grpc
from concurrent import futures
import time

from app.proto import document_search_pb2
from app.proto import document_search_pb2_grpc
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class DocumentSearchService(document_search_pb2_grpc.DocumentSearchServiceServicer):
    def __init__(self):
        self.model_name = EMBEDDING_MODEL
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        self.vectorstore = FAISS.load_local(vectorstore_dir, self.embeddings, allow_dangerous_deserialization=True)

    def RetrieveDocuments(self, request, context):
        query = request.query
        k = request.k or 10
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
