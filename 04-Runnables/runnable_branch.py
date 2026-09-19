from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableSequence,
    RunnableLambda,
    RunnableBranch
)


# Load Environment Variables

load_dotenv()


# Hugging Face LLM

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=500
)

model = ChatHuggingFace(llm=llm)


# Prompt 1

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)


# Prompt 2

prompt2 = PromptTemplate(
    template="Summarize the following text:\n{text}",
    input_variables=["text"]
)


# Output Parser

parser = StrOutputParser()


# Report Generation Chain

report_gen_chain = RunnableSequence(
    prompt1,
    model,
    parser
)


# Summary Chain

summary_chain = RunnableSequence(
    RunnableLambda(lambda x: {"text": x}),
    prompt2,
    model,
    parser
)


# Branch Chain

branch_chain = RunnableBranch(
    (
        lambda x: len(x.split()) > 500,
        summary_chain
    ),
    RunnablePassthrough()
)


# Final Chain

final_chain = RunnableSequence(
    report_gen_chain,
    branch_chain
)


# Run Final Chain

result = final_chain.invoke({
    "topic": "Russia vs Ukraine"
})


print(result)