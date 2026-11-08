
# Mini Project: Currency Converter using LangChain

import os
import requests
from dotenv import load_dotenv
from typing import Annotated

from langchain_groq import ChatGroq
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage

load_dotenv()


# Tool Creation

# Tool 1:
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    Fetch the currency conversion factor between a given
    base currency and a target currency.
    """

    api_key = os.getenv("EXCHANGERATE_API_KEY")

    if not api_key:
        raise ValueError("EXCHANGERATE_API_KEY is missing from .env")

    base_currency = base_currency.upper().strip()
    target_currency = target_currency.upper().strip()

    url = (
        f"https://v6.exchangerate-api.com/v6/"
        f"{api_key}/pair/{base_currency}/{target_currency}"
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()

    if data.get("result") != "success":
        raise ValueError(f"ExchangeRate API error: {data.get('error-type')}")

    return float(data["conversion_rate"])


# Tool 2:
@tool
def convert(
    base_currency_value: float,
    conversion_rate: Annotated[float, InjectedToolArg]
) -> float:
    """
    Convert a given amount in base currency to the target currency
    using the provided conversion rate.
    """
    return base_currency_value * conversion_rate


# Test the Tools
result = get_conversion_factor.invoke({
    "base_currency": "USD",
    "target_currency": "INR"
})

print("Conversion Factor:", result)

conversion_result = convert.invoke({
    "base_currency_value": 10,
    "conversion_rate": result
})

print("Converted Amount:", conversion_result)


# Tool Binding
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

llm_with_tools = llm.bind_tools([
    get_conversion_factor,
    convert
])


# Tool Calling
messages = [
    HumanMessage(
        content=(
            "What is the conversion factor between USD and INR? "
            "First call get_conversion_factor, then use convert "
            "to convert 10 USD into INR. Do not calculate manually."
        )
    )
]

# First LLM Call
ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

print("\nFirst Tool Calls:", ai_message.tool_calls)


# Tool Execution - Tool 1
for tool_call in ai_message.tool_calls:

    if tool_call["name"] == "get_conversion_factor":

        tool_message1 = get_conversion_factor.invoke(tool_call)

        conversion_rate = float(tool_message1.content)

        messages.append(tool_message1)

        print("\nConversion Rate:", conversion_rate)


# Second LLM Call
ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

print("\nSecond Tool Calls:", ai_message.tool_calls)


# Tool Execution - Tool 2
for tool_call in ai_message.tool_calls:

    if tool_call["name"] == "convert":

        tool_call_with_args = {
            **tool_call,
            "args": {
                **tool_call["args"],
                "conversion_rate": conversion_rate
            }
        }

        tool_message2 = convert.invoke(tool_call_with_args)

        messages.append(tool_message2)

        print("\nConverted Amount:", tool_message2.content)


# Final LLM Call
final_response = llm_with_tools.invoke(messages)
messages.append(final_response)

print("\nFinal AI Response:", final_response.content)


# Display Chat History
print("\nChat History:")
for message in messages:
    print(message)
