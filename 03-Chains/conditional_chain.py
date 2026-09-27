# ---------------------------------------------------------------------------
# conditional_chain.py
#
# WHAT THIS FILE DEMONSTRATES
# ---------------------------------------------------------------------------
# The last of the four chain types: RunnableBranch is if/elif/else, built
# out of Runnables. It takes (condition, chain) pairs plus one final
# fallback with no condition -- classify feedback as Positive or Negative,
# then run a DIFFERENT chain depending on which one it was.
#
# One import worth calling out, since it looks identical to something that
# broke two folders ago: PydanticOutputParser here comes from
# langchain_core.output_parsers, and I checked -- that import genuinely
# works on the current library. It's only the OLDER langchain.output_parsers
# path (used back in that other file) that was removed. Same class name,
# different package, only one of them is actually broken.
#
# Also worth noticing: x.sentiment (attribute access, not x["sentiment"])
# in the branch conditions works because parser2 is a PydanticOutputParser
# -- its output is a real Feedback OBJECT, not a plain dict. That's the
# same distinction from the with_structured_output lessons, showing up
# again here.
# ---------------------------------------------------------------------------

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# with the help of Runnable, we can execute multiple chains at once.
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.3,
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()


class Feedback(BaseModel):
    sentiment: Literal['Positive', 'Negative'] = Field(description='Give the sentiment of the feedback')


parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into Positive or Negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

# "classifier_chain": fixed a small typo from the original ("classfier_chain")
classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='write an appropriate response for this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='write an appropriate response for this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    # (condition, chain)
    (lambda x: x.sentiment == 'Positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'Negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not find Sentiment")
)

chain = classifier_chain | branch_chain

try:
    result = chain.invoke({'feedback': 'this is a beautiful phone'})
    print(result)

    # Worth knowing: this draws as a single "Branch" box, not an expanded
    # fork like RunnableParallel's diagram did. Which path actually runs is
    # a runtime decision based on the classified sentiment, not something
    # fixed in the chain's static structure -- so there's less for a
    # structural diagram to show here than there was for Parallel.
    chain.get_graph().print_ascii()

except Exception as e:
    # This is the real safety net here, more so than the RunnableBranch
    # fallback below. Feedback.sentiment is a strict Literal, so if the
    # model doesn't return exactly "Positive" or "Negative", parser2 raises
    # a validation error at the classification step itself -- before
    # branch_chain ever runs. The "Could not find Sentiment" fallback
    # would only fire if sentiment held some OTHER already-valid value
    # that just isn't Positive or Negative, which the strict Literal
    # doesn't actually allow through. This except block is what actually
    # catches a real classification failure.
    print(f"Something went wrong running the chain: {e}")