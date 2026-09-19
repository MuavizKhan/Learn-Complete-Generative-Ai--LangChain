from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser




load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)



# First prompt → Generate detailed report

template1 = PromptTemplate(
    input_variables=["topic"],
    template="Write a detailed report on {topic}"
)


# Second prompt → Summarize the report

template2 = PromptTemplate(
    input_variables=["text"],
    template="Write a summary of the following text:\n{text}"
)



parser = StrOutputParser()

# using CHAIN's
chain1 = template1 | model | parser


result = chain1.invoke({
    "topic": "Black Hole"
})

chain2 = template2 | model | parser

result1 = chain2.invoke({
    "text": result
})


print("\n========== FINAL SUMMARY ==========\n")

print(result1)