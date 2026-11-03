
from langchain_core.tools import tool


# ==========================================
# Custom Tools
# ==========================================

@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


# ==========================================
# Create a Custom Toolkit
# ==========================================

class MathToolkit:

    def get_tools(self):
        return [add, multiply]


# ==========================================
# Initialize Toolkit and Get Tools
# ==========================================

toolkit = MathToolkit()
tools = toolkit.get_tools()


# ==========================================
# Display Available Tools
# ==========================================

for tool in tools:
    print(tool.name, "=>", tool.description)
