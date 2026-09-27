# ---------------------------------------------------------------------------
# parallel_chain.py
#
# WHAT THIS FILE DEMONSTRATES
# ---------------------------------------------------------------------------
# The problem left over from sequential_chain.py: prompt3 needs TWO
# variables (notes and quiz), and a plain string can only auto-fill a
# template that needs exactly one. RunnableParallel is the actual fix --
# it runs several chains on the SAME input at once and returns a dict with
# one named key per chain, which is exactly the shape a multi-variable
# template needs: {"notes": ..., "quiz": ...} slots straight into
# prompt3's {notes} and {quiz}.
#
# "Parallel" isn't just a name for "organized side by side" here -- I
# measured it. Two branches that each individually take 1 second finished
# together in ~1.01 seconds through RunnableParallel, not ~2. The notes
# branch and the quiz branch actually run at the same time, not one after
# the other, which is the real reason to reach for this over two sequential
# calls.
#
# Also worth noticing: this file uses two different models -- model1
# (gpt-oss-120b) for the notes and the final merge, model2 (the free
# Llama-3.1-8B-Instruct from 01-Models) for the quiz. A chain isn't tied
# to one model; different branches can call different ones.
# ---------------------------------------------------------------------------

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()


llm1 = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.5,
)

model1 = ChatHuggingFace(llm=llm1)

llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.5,
)

model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answer from the following text \n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="Merge the provided note and quiz in a single document"
    " \n notes -> {notes} and quiz -> {quiz}",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

# making two chains, run at the same time -<
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

# Merging both the chains >-
merge_chain = prompt3 | model1 | parser

# final_chain
chain = parallel_chain | merge_chain

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

try:
    result = chain.invoke({'text': text})
    print(result)

    chain.get_graph().print_ascii()

except Exception as e:
    print(f"Something went wrong running the chain: {e}")