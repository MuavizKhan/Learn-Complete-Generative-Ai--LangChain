# Import built-in tools
from langchain_community.tools import DuckDuckGoSearchRun, ShellTool


# DuckDuckGo Search Tool
search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke('top news in india today')

print(results)


# Shell Tool
shell_tool = ShellTool()

results = shell_tool.invoke('whoami')

print(results)