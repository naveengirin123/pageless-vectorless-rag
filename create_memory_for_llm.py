
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# STEP 1 : Load Documents
def load_documents(data_path):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()

    # remove page dependency (pageless RAG)
    for doc in documents:
        doc.metadata.pop("page", None)

    return documents


documents = load_documents("data")


# STEP 2 : Pageless text splitting (semantic splitting)
def split_documents(docs):

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size = 800,          # bigger chunks = better context
        chunk_overlap = 150,

        separators=[
            "\n\n",
            "\n",
            ".",
            " ",
            ""
        ]

    )
    chunks = text_splitter.split_documents(docs)
    return chunks
text_chunks = split_documents(documents)


# STEP 3 : Create embeddings
def load_embeddings():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model
embedding_model = load_embeddings()


# STEP 4 : Store in FAISS
def create_vector_store(chunks, embedding_model):

    DB_PATH = "vectorstore/db_faiss"
    db = FAISS.from_documents(
        chunks,
        embedding_model
    )
    db.save_local(DB_PATH)
    print("Vector database created successfully")


create_vector_store(text_chunks, embedding_model)