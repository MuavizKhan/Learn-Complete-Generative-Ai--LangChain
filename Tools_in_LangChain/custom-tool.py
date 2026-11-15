# ---------------------------------------------------------------------------
# custom-tool.py  --  Tools in LangChain, part 2: Custom Tools
# ---------------------------------------------------------------------------
#
# WHAT A CUSTOM TOOL IS (your notes, p.121)
# A tool you define yourself. Reach for one when you want to:
#   - call your own APIs,
#   - wrap your own business logic,
#   - let an LLM work with your database, product or app.
#
# THE THREE WAYS TO BUILD ONE (your notes, p.122)
#   Method 1: the @tool decorator          (shortest)
#   Method 2: StructuredTool + Pydantic    (you control the input schema)
#   Method 3: a BaseTool subclass          (full control, most code)
#
# One refinement from testing, because it makes the three easier to hold in
# your head. The notes draw them as three side-by-side boxes, but they are
# really a ladder. I checked what @tool actually returns:
#       type(multiply)  ->  StructuredTool
#       StructuredTool is a subclass of BaseTool
# So Methods 1 and 2 build the SAME kind of object. @tool is shorthand that
# works out the name (from the function name), the description (from the
# docstring) and the input schema (from the type hints) for you. Method 2
# lets you spell those out yourself. BaseTool is the class underneath both,
# exactly as the notes say ("built on top of BaseTool").
#
# THE THREAD THAT TIES THIS TO built-in.py
# Whichever way you build it, a tool is three things the LLM reads:
# a NAME, a DESCRIPTION, and an ARGS schema. The LLM never sees your Python.
# So these three are the real "interface" you are designing.
#
# HEADS-UP: all three tools below are named "multiply" on purpose, so you can
# compare their outputs. If you ever give several tools to one model in a
# single list, give each a unique name -- the name is how the model asks for
# a tool.
# ---------------------------------------------------------------------------

from langchain_core.tools import tool, StructuredTool, BaseTool
from pydantic import BaseModel, Field
from typing import Type


# ==========================================
# Method 1 - Using @tool decorator
# ==========================================

# The docstring is NOT just documentation here -- it becomes the tool's
# description, i.e. the instruction the LLM reads to decide when to use it.
# I tested leaving it out:
#     ValueError: Function must have a docstring if description not provided.
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


# Invoke the tool
result = multiply.invoke({"a": 3, "b": 5})

print("Method 1 result:", result)

# Access tool properties
print(multiply.name)          # taken from the function name
print(multiply.description)   # taken from the docstring
print(multiply.args)          # built from the type hints

# Notice the args have a type for each input but NO description of what each
# input means. That is the one thing plain @tool doesn't give you for free.

# Access JSON schema
print(multiply.args_schema.model_json_schema())


# Method 1b - the same @tool, but with per-input descriptions.
# parse_docstring=True reads the "Args:" section of a Google-style docstring.
# I tested this: the result carries the same per-input descriptions that
# Methods 2 and 3 get from Pydantic.
@tool(parse_docstring=True)
def multiply_documented(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    return a * b


print("Method 1b args:", multiply_documented.args)


# ==========================================
# Method 2 - Using StructuredTool and Pydantic
# ==========================================

# Define input schema using Pydantic (the same Field(description=...) idea
# you met in pydantic_demo.py, now used to describe a tool's inputs)
class MultiplyInput(BaseModel):
    a: int = Field(..., description="The first number to multiply")
    b: int = Field(..., description="The second number to multiply")


# Create the function (a plain function -- no decorator, no docstring needed,
# because the description is passed in explicitly below)
def multiply_func(a: int, b: int) -> int:
    return a * b


# Convert function into a StructuredTool: here YOU supply the name,
# description and schema instead of having them inferred.
multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)


# Invoke the tool
result = multiply_tool.invoke({"a": 3, "b": 3})

print("Method 2 result:", result)

# Access tool properties
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)     # now each input carries its description


# ==========================================
# Method 3 - Using BaseTool class
# ==========================================

# Define argument schema using Pydantic
class MultiplyInputV3(BaseModel):
    a: int = Field(..., description="The first number to multiply")
    b: int = Field(..., description="The second number to multiply")


# Create a custom tool class: name, description and args_schema are class
# attributes, and the actual work goes in _run().
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

print("Method 3 result:", result)

# Access tool properties
print(multiply_base_tool.name)
print(multiply_base_tool.description)
print(multiply_base_tool.args)


# ==========================================
# Compare the three side by side
# ==========================================

# Methods 1 and 2 should report the same class -- the "ladder" from the
# header, visible in your own output.
print("\nWhat each method actually built:")
for label, t in (("Method 1", multiply),
                 ("Method 2", multiply_tool),
                 ("Method 3", multiply_base_tool)):
    print(f"  {label}: {type(t).__name__}")


# ==========================================
# Bonus - tools check what an LLM sends them
# ==========================================

# An LLM fills in a tool's arguments itself, and it doesn't always get them
# right. Every tool validates its input against its args schema first --
# the same Pydantic behaviour from pydantic_demo.py. I tested three inputs
# on the Method 1 tool:
#     {"a": "3", "b": 5}      -> 15           (the string "3" is coerced to 3)
#     {"a": "three", "b": 5}  -> ValidationError (cannot be read as an integer)
#     {"a": 3}                -> ValidationError (required input "b" is missing)
# Coercing when it can, and failing loudly when it can't, is what you want
# between a model and real code.
print("\nValidation demo:")
for payload in ({"a": "3", "b": 5}, {"a": "three", "b": 5}, {"a": 3}):
    try:
        print(f"  {payload} -> {multiply.invoke(payload)}")
    except Exception as e:
        print(f"  {payload} -> {type(e).__name__}")