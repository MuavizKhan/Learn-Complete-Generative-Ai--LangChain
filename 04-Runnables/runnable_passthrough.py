from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough
)


# Load Environment Variables

load_dotenv()


# Prompt 1

prompt1 = PromptTemplate(
    template="Write a short joke about {topic}",
    input_variables=["topic"]
)


# Prompt 2

prompt2 = PromptTemplate(
    template="Explain the following joke:\n\n{text}",
    input_variables=["text"]
)


# Hugging Face LLM

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=50,
    temperature=0.7
)


# Chat Model

model = ChatHuggingFace(
    llm=llm
)


# Output Parser

parser = StrOutputParser()


# Model Test

test_response = model.invoke(
    "Write one short joke about cricket."
)

print("Model response:")
print(test_response.content)


# Joke Generation Chain

joke_gen_chain = RunnableSequence(
    prompt1,
    model,
    parser
)


# Parallel Chain

parallel_chain = RunnableParallel({

    "joke": RunnablePassthrough(),

    "explanation": RunnableSequence(
        {
            "text": RunnablePassthrough()
        },
        prompt2,
        model,
        parser
    )
})


# Final Chain

final_chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain
)


# Run Final Chain
print("Starting final chain...")

result = final_chain.invoke({
    "topic": "cricket"
})

print("Final chain completed.")

print("JOKE:")
print(result["joke"])


print("\nEXPLANATION:")
print(result["explanation"])