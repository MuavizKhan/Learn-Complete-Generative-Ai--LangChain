from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# chat template
chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful customer support assistant.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = [] 
# load chat history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())


print(chat_history)


# create our prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'What is the status of my order?'})

print(prompt)