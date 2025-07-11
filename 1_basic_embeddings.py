from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.embeddings.create(
    input="Apple is a company in San Francisco", model="text-embedding-3-large"
)

print(response.data[0].model_dump_json())
