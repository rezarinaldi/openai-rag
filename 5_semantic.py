from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

with open("markdown.txt", "r") as f:
    text = f.read()

splitter = SemanticChunker(OpenAIEmbeddings())

documents = splitter.create_documents([text])
print(documents)
