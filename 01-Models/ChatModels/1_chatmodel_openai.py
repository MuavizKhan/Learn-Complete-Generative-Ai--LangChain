from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-5.6-luna"
)

result = model.invoke("What is the Capital of India?")

print(result.content)