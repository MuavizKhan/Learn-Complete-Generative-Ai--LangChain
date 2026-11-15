# ---------------------------------------------------------------------------
# built-in.py  --  Tools in LangChain, part 1: Built-in Tools
# ---------------------------------------------------------------------------
#
# THE BIG IDEA (your notes, "What is a Tool?")
# An LLM is great at thinking and writing, but it cannot, by itself, look up
# today's news, run code, or touch your machine. A TOOL is just a Python
# function packaged so an LLM can understand it and ask for it to be called.
# The notes' overview diagram shows where this module sits:
#
#       Tools  ->  Tool Calling (tool + LLM)  ->  Agents
#
# This folder is the first box. Nothing here talks to an LLM yet -- we are
# only learning what a tool IS and how to run one by hand.
#
# WHAT "BUILT-IN" MEANS (your notes)
# A built-in tool is one LangChain already wrote for you: import it and use
# it, no function logic to write. This file uses two of the eight listed in
# your notes:
#     DuckDuckGoSearchRun  -> web search            (READS the outside world)
#     ShellTool            -> run shell commands    (ACTS on your machine)
# That read-vs-act difference matters a lot -- see the safety note below.
#
# BEFORE YOU RUN THIS: two extra installs (I hit both errors myself)
#     pip install ddgs                  # DuckDuckGoSearchRun fails on creation
#                                       # without it: "Could not import ddgs"
#     pip install langchain-experimental  # ShellTool fails on creation without
#                                       # it: "BashProcess has been moved to
#                                       # langchain experimental"
# Also: importing from langchain_community prints a "being sunset" warning.
# It still works today -- same situation as the document loaders earlier.
# ---------------------------------------------------------------------------

# Import built-in tools
from langchain_community.tools import DuckDuckGoSearchRun, ShellTool


# DuckDuckGo Search Tool  (needs internet; no API key required)
search_tool = DuckDuckGoSearchRun()

# Shell Tool
#
# SAFETY NOTE -- the most important comment in this file.
# ShellTool lets whoever calls it run commands on YOUR computer. Today that
# caller is you, typing 'whoami'. Once a tool is handed to an LLM agent
# (two modules from now), the caller becomes a model deciding on its own --
# and text it read from a web page or a document could steer it. Treat this
# tool as dangerous by default.
#
# LangChain ships a safeguard for exactly this: ask_human_input=True prints
# each command and waits for you to type y/n before running it. I tested it
# -- answering "n" means the command is skipped and the tool returns None.
# It's switched on here so the safe pattern is the one you copy. For an
# unattended demo you can set it to False, but only with harmless commands.
shell_tool = ShellTool(ask_human_input=True)


# STEP 1 -- see what an LLM would actually "see" about each tool.
# An LLM never sees your Python code. It sees only three things: a NAME, a
# DESCRIPTION, and the shape of the INPUT (the args schema). This is the
# "packaged in a way the LLM can understand" part of your notes' definition.
# I printed these two from the real tools:
#   duckduckgo_search -> "...Useful for when you need to answer questions
#                         about current events. Input should be a search query."
#                        input: query (a string)
#   terminal          -> "Run shell commands on this <your OS> machine."
#                        input: commands (a string, or a list of strings)
# (The shell tool builds its description from your operating system, so on
# Windows it will say Windows -- my sandbox said Linux.)
for tool_obj in (search_tool, shell_tool):
    print(f"NAME:        {tool_obj.name}")
    print(f"DESCRIPTION: {tool_obj.description}")
    print(f"ARGS:        {tool_obj.args}")
    print()


# STEP 2 -- run each tool by hand with .invoke().
# Tools are Runnables, just like retrievers, prompts and models -- I checked:
# BaseTool is a subclass of Runnable. So the .invoke() you already know works
# on tools too, and tools can later be dropped into chains.
try:
    results = search_tool.invoke('top news in india today')
    print(results)
except Exception as e:
    # A live web call can fail (no internet, rate limits). Don't let that
    # stop the rest of the file from running.
    print(f"Search tool failed: {e}")

print()

try:
    results = shell_tool.invoke('whoami')
    if results is None:
        print("Command was not run (you answered 'n' to the approval prompt).")
    else:
        print(results)
except Exception as e:
    print(f"Shell tool failed: {e}")