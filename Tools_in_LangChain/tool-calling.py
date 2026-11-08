
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

# Custom Tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# Initialize LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

llm_with_tools = llm.bind_tools([multiply])

# Human Message
messages = [HumanMessage(content="What is the product of 3 and 4?")]

# AI Message
response = llm_with_tools.invoke(messages)
messages.append(response)

# Tool Execution and Tool Message
for tool_call in response.tool_calls:
    result = multiply.invoke(tool_call["args"])
    messages.append(
        ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"],
            name=tool_call["name"]
        )
    )

# Final AI Message
final_response = llm_with_tools.invoke(messages)
messages.append(final_response)

print("AI:", final_response.content)

# Chat History
for message in messages:
    print(f"{message.type}: {message.content or message.tool_calls}")
