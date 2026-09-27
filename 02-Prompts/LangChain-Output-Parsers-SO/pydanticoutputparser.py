from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

# Same root cause as the last file: langchain.output_parsers doesn't exist
# in current langchain (1.4.2) -- confirmed the same way, by actually
# importing it and reading the ModuleNotFoundError. Here, though, I kept
# PydanticOutputParser itself rather than swap to with_structured_output:
# this file's entire purpose IS demonstrating this specific parser, so
# replacing it would mean teaching a different thing under the same
# filename. The fix that keeps the lesson intact is just correcting where
# it's imported from -- langchain_classic (pip install langchain-classic),
# which still ships both this parser and ResponseSchema unchanged.
from langchain_classic.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)


class Person(BaseModel):

    name: str = Field(
        description="Name of the person"
    )

    # ge (greater-or-equal), not gt (strictly greater than): gt=18 quietly
    # excludes anyone who is exactly 18, which is an odd way to say "adult."
    age: int = Field(
        ge=18,
        description="Age of the person"
    )

    city: str = Field(
        description="Name of the city the person belongs to"
    )


parser = PydanticOutputParser(
    pydantic_object=Person
)


template = PromptTemplate(
    template=(
        "Generate the name, age and city of a fictional "
        "{place} person.\n"
        "{format_instruction}"
    ),
    input_variables=["place"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

chain = template | model | parser

try:
    final_result = chain.invoke({'place': 'kuwait'})
    print(final_result)

except Exception as e:
    print(f"Something went wrong parsing the model's output: {e}")