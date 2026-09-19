from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.8,
)

model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content="You are a helpful AI assistant."),
]

#static prompt
while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content)) 
    print("AI: ", result.content)


print(chat_history)