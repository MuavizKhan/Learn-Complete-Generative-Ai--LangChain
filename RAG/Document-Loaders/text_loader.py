from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=500
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = 'write a summary for the following poem - \n {poem}',
    input_variables = ['poem']
)

parser = StrOutputParser()

loader = TextLoader(r'C:\Users\Muaviz Khan\OneDrive\Desktop\GenAI--LangChain\RAG\Document-Loaders\cricket.txt', encoding='utf-8')

docs = loader.load()

print(docs)
#print(type(docs))
#print(len(docs))
#print(docs[0])
#print(type(docs[0]))
print(docs[0].page_content)
#print(docs[0].metadata)

chain = prompt | model | parser

print(chain.invoke({'poem':docs[0].page_content}))