from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# with the help of Runnable, we can execute multiple chains at once.
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=512,
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['Positive', 'Negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feedback text into Positive or Negative \n {feedback} \n {format_instruction}',
    input_variables = ['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classfier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = 'write an appropriate response for this positive feedback \n {feedback}',
    input_variables = ['feedback']
)

prompt3 = PromptTemplate(
    template = 'write an appropriate response for this negative  feedback \n {feedback}',
    input_variables = ['feedback']
)

branch_chain = RunnableBranch(
    # (condition, chain)
    (lambda x:x.sentiment == 'Positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'Negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not find Sentiment")
)

chain = classfier_chain | branch_chain

print(chain.invoke({'feedback':'this is a beautiful phone'}))

chain.get_graph().print_ascii()