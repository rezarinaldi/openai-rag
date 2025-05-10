from mistralai import Mistral
from dotenv import load_dotenv
from openai import OpenAI
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

openai_client = OpenAI()

chroma_client = chromadb.HttpClient("localhost", 8010)

client = Mistral(api_key=MISTRAL_API_KEY)

openai_ef = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
)

# ocr_response = client.ocr.process(
#     model="mistral-ocr-latest",
#     document={
#         "type": "document_url",
#         "document_url": "https://www.hasbro.com/common/instruct/00009.pdf"
#     }
# )
#
# content = ocr_response.model_dump()
#
# markdown_content = ""
#
# for page in content.get("pages", []):
#     if "markdown" in page:
#         markdown_content += page["markdown"] + "\n\n"
#
# splitter = SemanticChunker(OpenAIEmbeddings())
# documents = splitter.create_documents([markdown_content])

# chroma_client.delete_collection("monopoly-guide")
# collection = chroma_client.create_collection("monopoly-guide", embedding_function=openai_ef)
#
# collection.add(
#     documents=[doc.model_dump().get("page_content") for doc in documents],
#     metadatas=[{ "project_id": .... }]
#     ids=[str(i) for i in range(len(documents))]
# )

collection = chroma_client.get_collection(
    "monopoly-guide", embedding_function=openai_ef
)

result = collection.query(
    query_texts=["What is player role of banker?"],
    n_results=3,
    include=["distances", "documents"],
)

response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": f"Your job is to answer based on the provided context only! This is the context: {result.get('documents')}",
        },
        {"role": "user", "content": "What is player role of banker?"},
    ],
)

content = response.choices[0].message.content
print(content)
