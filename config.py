from dotenv import find_dotenv, load_dotenv
import os

_ = load_dotenv(find_dotenv())

# Zero-Shot Prompting
standard_prompt = '''Answer the following question with clear sentences. 
Avoid elaboration or unnecessary details.
Question: {question}
Answer:'''


# Zero-shot Chain-of-Thought Prompting
cot_prompt = '''Answer the following question using a step-by-step reasoning approach. 
Follow these guidelines:
- Identify the main entities in the question.
- Determine their relationships and relevant knowledge.
- Conclude with concise sentences that directly answers the question.

Question: {question}
Answer:'''

instruction = "You are an advanced model specializing in question answering."

prompt_list = {"standard_prompt": standard_prompt, "cot_prompt": cot_prompt}

# Zero-Shot Prompting
standard_prompt_african = '''Answer the following question with clear sentences. 
Avoid elaboration or unnecessary details.
Focus only on the African wild context when answering.

Question: {question}
Answer:'''


prompt_african_list = {"standard_prompt": standard_prompt_african}


# Zero-Shot Prompting
standard_prompt_african_onto = '''Answer the following question with clear sentences. 
Avoid elaboration or unnecessary details.

Question: {question}
Answer:'''

onto_instruction = """You are an advanced model specializing in question answering. 

Use the following African wild ontologh:

{ontology}
"""


prompt_african_list_onto = {"standard_prompt": standard_prompt_african_onto}


models_list = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E",

    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",

    "mistralai/Ministral-8B-Instruct-2410",

    "mistralai/Mistral-7B-Instruct-v0.3",

    "tiiuae/Falcon3-7B-Instruct",

    "tiiuae/Falcon3-Mamba-7B-Instruct",
]

huggingface_token = os.environ['HUGGINGFACE_ACCESS_TOKEN']
openai_token = os.environ['OPENAI_KEY']
