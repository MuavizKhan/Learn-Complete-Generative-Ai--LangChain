import random
from abc import ABC, abstractmethod

# Component 1 -> Dummy LLM Class
# class NakliLLM:

#     def __init__(self):
#         print("LLM Created")

#     def predict(self, prompt):

#         response_list = [
#             'Delhi is the capital of India',
#             'IPL is a cricket league',
#             'AI stands for Artificial Intelligence'
#         ]

#         return {'response': random.choice(response_list)}


# llm = NakliLLM()

# print(llm.predict('What is the capital of India'))


# Component 2 -> Prompt Template
# class NakliPromptTemplate:

#     def __init__(self, template, input_variables):
#         self.template = template          # <-- removed comma
#         self.input_variables = input_variables

#     def format(self, input_dict):
#         return self.template.format(**input_dict)


# template = NakliPromptTemplate(
#     template='write a {length} poem about {topic}',
#     input_variables=['length','topic']
# )

# prompt = template.format({'length':'short', 'topic': 'India'})
# print(prompt)

# llm = NakliLLM()
# print(llm.predict(prompt))



# component 3 -> Nakli LLM-Chain
# job i to conect 1<->2


class NakliLLMChain:

    def __init__(self,llm, prompt):
        self.llm = llm
        self.prompt = prompt

# not flexible as unable to hit multiple call on the LLM.
# not flexible to create any kind of workflows.
    def run(self, input_dict):

        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)

        return result['response']

# chain = NakliLLMChain(llm, template)
# chain.run({'length':'short', 'topic':'India'})


# now we want to standardize the components, as 'NakliPromptTemplate' requires 'format' function 
# and 'NakliLLM' requires 'predict' function.

# HOW? -> make both th classes into RUNNABLES & common methods must be present in all the RUNNABLES.
# using 'Abstraction'

class Runnable(ABC):

    @abstractmethod
    def invoke(self, input_data):
        pass


class NakliLLM(Runnable):

    def __init__(self):
        print("LLM Created")

    def invoke(self, prompt):
        response_list = [
                'Delhi is the capital of India',
                'IPL is a cricket league',
                'AI stands for Artificial Intelligence'
                ]
        
        return {'response': random.choice(response_list)}

    def predict(self, prompt):

        response_list = [
            'Delhi is the capital of India',
            'IPL is a cricket league',
            'AI stands for Artificial Intelligence'
        ]

        return "This method is going to get deprecated in the future. Use 'invoke' Instead."


class NakliPromptTemplate(Runnable):

    def __init__(self, template, input_variables):
        self.template = template          # <-- removed comma
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)


    def format(self, input_dict):
        return "This method is going to get deprecated in the future. Use 'invoke' Instead." 


class NakliStrOUtputParser(Runnable):
    def __init__(self):
        pass

    def invoke(self, input_data):
        return input_data['response']


# Now we will form Chains - using these components.
class RunnableConnector(Runnable):

    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):

        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)

        return input_data

template = NakliPromptTemplate(
    template='write a {length} poem about {topic}',
    input_variables=['length', 'topic']
)

llm = NakliLLM()

parser = NakliStrOUtputParser()

chain = RunnableConnector([template, llm, parser])

print("Chain created")

result = chain.invoke({
    'length': 'long',
    'topic': 'India'
})

print(result)

template1 = NakliPromptTemplate(
    template = 'Write a joke about {topic}',
    input_variables = ['topic']
)

template2 = NakliPromptTemplate(
    template = "Explain the following joke {response}",
    input_variables = ['response']
)

llm = NakliLLM()
parser = NakliStrOUtputParser()

# print JOKE
chain1 = RunnableConnector([template1, llm])

# print SUMMARY
chain2 = RunnableConnector([template2, llm, parser])

final_chain = RunnableConnector([chain1, chain2])
final_chain.invoke({'topic':'cricket'})