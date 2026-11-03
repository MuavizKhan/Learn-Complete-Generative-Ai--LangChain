
from langchain_core.tools import tool, StructuredTool, BaseTool
from pydantic import BaseModel, Field
from typing import Type


# ==========================================
# Method 1 - Using @tool decorator
# ==========================================

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


# Invoke the tool
result = multiply.invoke({"a": 3, "b": 5})

print(result)

# Access tool properties
print(multiply.name)
print(multiply.description)
print(multiply.args)

# Access JSON schema
print(multiply.args_schema.model_json_schema())


# ==========================================
# Method 2 - Using StructuredTool and Pydantic
# ==========================================

# Define input schema using Pydantic
class MultiplyInput(BaseModel):
    a: int = Field(..., description="The first number to multiply")
    b: int = Field(..., description="The second number to multiply")


# Create the function
def multiply_func(a: int, b: int) -> int:
    return a * b


# Convert function into a StructuredTool
multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)


# Invoke the tool
result = multiply_tool.invoke({"a": 3, "b": 3})

print(result)

# Access tool properties
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)


# ==========================================
# Method 3 - Using BaseTool class
# ==========================================

# Define argument schema using Pydantic
class MultiplyInputV3(BaseModel):
    a: int = Field(..., description="The first number to multiply")
    b: int = Field(..., description="The second number to multiply")


# Create a custom tool class
class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"

    args_schema: Type[BaseModel] = MultiplyInputV3

    def _run(self, a: int, b: int) -> int:
        return a * b


# Create an instance of the tool
multiply_base_tool = MultiplyTool()


# Invoke the tool
result = multiply_base_tool.invoke({"a": 3, "b": 3})

print(result)

# Access tool properties
print(multiply_base_tool.name)
print(multiply_base_tool.description)
print(multiply_base_tool.args)
