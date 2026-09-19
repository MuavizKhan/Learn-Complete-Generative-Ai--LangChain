from langchain_community.document_loaders import WebBaseLoader
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
    template = 'Answer the following question \n {question} from the following text \n {text}',
    input_variables = ['question','text']
)

parser = StrOutputParser()


url = 'https://www.apple.com/in/shop/buy-iphone/iphone-18-pro/6.9%22-display-1tb-burgundy?afid=p240%7Cgo~cmp-11188658057~adg-199581584786~ad-824134965312_pla-2501539316395~dev-c~ext-~prd-MJRR4HN%2FA-IN~mca-5055540~nt-search&cid=aos-in-kwgo-pla-iphone-iphone--product-MJRR4HN%2FA-IN'
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser
print(chain.invoke({'question':'what is the product that we are talking about?','text':docs[0].page_content}))