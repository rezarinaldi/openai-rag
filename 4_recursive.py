from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("markdown.txt", "r") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,  # 15-20% of chunk_size
)

documents = splitter.create_documents([text])
print(documents)
