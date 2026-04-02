import os

from transformers import pipeline

from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import PromptTemplate

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS


# STEP 1 : Load ENV
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


# STEP 2 : Load LLM (FREE local model)
HF_MODEL = os.environ.get("HF_MODEL")


def load_llm():

    pipe = pipeline(

        "text-generation",

        model = HF_MODEL,

        max_new_tokens = 256,

        temperature = 0.3

    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return llm


# STEP 3 : Prompt

CUSTOM_PROMPT = """

Use the context to answer the question.

If answer not in context say:
"I don't know"

Context:
{context}

Question:
{question}

Answer:

"""


def set_prompt():

    prompt = PromptTemplate(

        template = CUSTOM_PROMPT,

        input_variables = ["context","question"]

    )

    return prompt


# STEP 4 : Load Vector DB

DB_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)


# STEP 5 : Create RAG chain

chain = RetrievalQA.from_chain_type(

    llm = load_llm(),

    chain_type = "stuff",

    retriever = db.as_retriever(

        search_kwargs = {"k":4}   # slightly better retrieval

    ),

    return_source_documents = True,

    chain_type_kwargs = {

        "prompt": set_prompt()

    }

)


# STEP 6 : Query loop

while True:

    user_query = input("\nAsk question (type exit): ")

    if user_query == "exit":
        break

    response = chain.invoke({

        "query": user_query

    })

    print("\nAnswer:")
    print(response["result"])

    print("\nSources:")
    print(response["source_documents"])