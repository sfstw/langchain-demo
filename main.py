import os
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.schema.messages import SystemMessage
from langchain.agents.agent_toolkits import (
    create_conversational_retrieval_agent,
    create_retriever_tool,
)

data_directory = 'data'

embeddings_deployment = "text-embedding"
openai_api_version = "2023-07-01-preview"
llm_deployment = "gpt-35-turbo-16k"  # "gpt-35-turbo-16k" "gpt-4" "gpt-35-turbo"

metadata_field_info = [
    AttributeInfo(
        name="source file",
        description="The name of the pdf file",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="source page",
        description="The page in the source pdf file",
        type="integer",
    ),
]

system_message = SystemMessage(
    content=(
        "You are a helpful expert on Steadforce company policies. "
        "Use your tools to search information and answer the questions in German."
        "If you cannot find relevant information using your tools, say that you don't know the answer."
        "Answer in German."
    )
)

alternative_system_message = SystemMessage(
    content=(
        "Use your tools to look up the information you need to answer the questions in German. "
        "If you can't find relevant information using your tools, say that you don't know the answer."
    )
)


def main():
    # documents = load_documents()

    vector_db = load_vector_db()

    llm = init_llm()

    self_query_retriever = init_retriever(llm, vector_db)

    retriever_tool = create_retriever_tool(
        self_query_retriever,
        name="search_steadforce_company_policies",
        description="Searches and returns company policies of Steadforce (SF).",
    )

    conversational_retrieval_agent = create_conversational_retrieval_agent(
        llm=llm,
        tools=[retriever_tool],
        system_message=system_message,
        verbose=True,
    )

    prompt = ''

    while prompt != 'quit':
        prompt = input("Wie kann ich helfen?\n")
        conversational_retrieval_agent.invoke({"input": prompt})


def load_documents():
    documents = []
    for page_name in os.listdir(data_directory):
        subdir_path = os.path.join(data_directory, page_name)

        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                try:
                    if file.endswith('.pdf'):
                        pdf_path = os.path.join(subdir_path, file)
                        loader = PyPDFLoader(pdf_path, extract_images=True)
                        for loaded_document in loader.load():
                            documents.append(loaded_document)
                            print(f"File loaded: {file}")
                    else:
                        print(f"File skipped because it is not a PDF file: {file}")
                except Exception as e:
                    print(f"File {file} failed with Exception: {e}")
    return documents


def load_documents_into_vector_db(documents):
    embeddings_fn = AzureOpenAIEmbeddings(
        openai_api_version=openai_api_version,
        azure_deployment=embeddings_deployment,
    )
    client = chromadb.HttpClient(settings=Settings(allow_reset=True))
    client.reset()
    collection_name = "Steadyhome"
    return Chroma.from_documents(
        documents,
        embeddings_fn,
        client=client,
        collection_name=collection_name,
    )


def load_vector_db():
    embeddings_fn = AzureOpenAIEmbeddings(
        openai_api_version=openai_api_version,
        azure_deployment=embeddings_deployment,
    )
    client = chromadb.HttpClient(settings=Settings(allow_reset=True))
    collection_name = "Steadyhome"
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings_fn,
        client=client,
    )


def init_llm():
    return AzureChatOpenAI(
        azure_deployment=llm_deployment,
        openai_api_version=openai_api_version,
    )


def init_retriever(llm, vector_db):
    return SelfQueryRetriever.from_llm(
        llm,
        vector_db,
        document_contents="Information for employees of Steadforce (SF)",
        metadata_field_info=metadata_field_info,
        verbose=True,
    )


if __name__ == "__main__":
    main()
