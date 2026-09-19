from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4.1-Flash",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = 'Generate a tweet about {topic}',
    input_variables= ['topic']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(lambda x: len(x.split()))
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic':'AI'})

final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)