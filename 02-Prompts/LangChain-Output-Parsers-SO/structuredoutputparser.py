from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()


# LLM
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=512,
)

model = ChatHuggingFace(llm=llm)


# Output Schema
schema = [
    ResponseSchema(
        name="fact_1",
        description="Fact 1 about the topic"
    ),
    ResponseSchema(
        name="fact_2",
        description="Fact 2 about the topic"
    ),
    ResponseSchema(
        name="fact_3",
        description="Fact 3 about the topic"
    ),
]


# Output Parser
parser = StructuredOutputParser.from_response_schemas(schema)


# Prompt
template = PromptTemplate(
    template="Give 3 facts about {topic}\n{format_instruction}",
    input_variables=["topic"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)


# Chain
chain = template | model | parser


# Invoke
result = chain.invoke({
    "topic": "black hole"
})


print(result)