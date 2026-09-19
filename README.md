# Learn Complete Generative AI — LangChain

A hands-on, beginner-to-advanced path through Generative AI with LangChain — built module by module while learning, so anyone can clone this repo and follow the same path from scratch.

## How to use this repo

1. Clone it and set up a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and drop in your own API keys.
4. Work through the module folders in order (01, 02, 03...) — each one builds on the last.

## Roadmap

| # | Module | Status | Covers |
|---|--------|--------|--------|
| 01 | [Models](./01-Models) | ✅ | Chat models, embedding models, LLMs — OpenAI, Anthropic, Gemini, Hugging Face |
| 02 | [Prompts](./02-Prompts) | ✅ | Prompt templates, chat history, output parsers, structured output |
| 03 | [Chains](./03-Chains) | 🔶 | Simple, sequential, parallel, and conditional chains |
| 04 | [Runnables](./04-Runnables) | 🔶 | LCEL, RunnableSequence, RunnableParallel, RunnableBranch, RunnableLambda, RunnablePassthrough |
| 05 | Memory | ⬜ | LangGraph persistence & checkpointing |
| 06 | Indexes / Retrieval | ⬜ | Embeddings, vector databases, retrievers |
| 07 | RAG | ⬜ | Full retrieval-augmented generation pipeline |
| 08 | Agents | ⬜ | Tool use, function calling, MCP |
| 09 | Evaluation & Guardrails | ⬜ | Testing, RAGAS, safety |

*(✅ done · 🔶 in progress · ⬜ planned — will keep pushing whatever i have learned)*

## Setup

```bash
git clone https://github.com/MuavizKhan/Learn-Complete-Generative-Ai--LangChain.git
cd Learn-Complete-Generative-Ai--LangChain

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
copy .env.example .env       # Windows — or: cp .env.example .env
```

Then open `.env` and add your own keys.

## Notes

This is a personal learning log — new modules get added as they're learned. Follow along, fork it, or open an issue if a beginner explanation could be clearer.
