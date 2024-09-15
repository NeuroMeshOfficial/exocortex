import pandas as pd
import sys
import re
from together import Together
from openai import OpenAI
import asyncio
import anthropic

# Constants and configuration
DEFAULT_OUTPUT_FILENAME = 'results_exo_cortex.txt'
TOGETHER_API_KEY = "key1"
OPENAI_API_KEY = "key2"
CLAUDE_API_KEY = "key3"



anthropic_client = anthropic.Anthropic(
    api_key=CLAUDE_API_KEY,
)
# Initialize clients
together_client = Together(api_key=TOGETHER_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Helper functions
def extract_integer_answer(response_content):
    try:
        return int(response_content)
    except ValueError:
        numbers = re.findall(r'\d+', response_content)
        return int(numbers[-1]) if numbers else None

def log_result(question, answer, model_response, filename, answers):
    with open(filename, 'a') as f:
        print(f"Question: {question[:80]}; Answer: {answer}; Model Response: {model_response}; Individual Answers: {answers}", file=f)
    print(f"Question: {question[:80]}; Answer: {answer}; Model Response: {model_response}; Individual Answers: {answers}")

# Main functions
def run_gpt4o(question, answer, output_filename):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that can answer questions and provide step-by-step solutions."},
            {"role": "user", "content": f"Please provide the steps required to solve the following question: {question} returning the solution in the end as an integer in the format of Answer: xxx."},
        ]
    )
    gpt4o_answer = extract_integer_answer(response.choices[0].message.content)
    
    if gpt4o_answer is not None:
        log_result(question, answer, gpt4o_answer, output_filename)
        return int(gpt4o_answer) == int(answer)
    else:
        print("Failed to convert GPT-4o response to an integer.")
        return False

async def run_exo_cortex(question, answer):
    models = [
        ("meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", together_client),
        ("mistralai/Mixtral-8x22B-Instruct-v0.1", together_client),
        ("gpt-4o", openai_client),
        ("claude-3-5-sonnet-20240620", anthropic_client)
    ]
    
    answers = []
    async def get_model_response(model, client, question):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question + " Please return the answer as an integer in the format Answer: xxx."}]
        ) if model != "claude-3-5-sonnet-20240620" else client.messages.create(
            model=model,
            max_tokens=1500,
            messages=[{"role": "user", "content": [
                {
                    "type": "text",
                    "text": question + " Please return the answer as an integer in the format Answer: xxx."
                }
            ]}]
        )
        
        answer = response.choices[0].message.content if model != "claude-3-5-sonnet-20240620" else response.content[0].text
        print(f"\n\n\n Answer generated for {model}: {answer}")
        return answer

    tasks = [get_model_response(model, client, question) for model, client in models]
    answers = await asyncio.gather(*tasks)
    integer_answers = [extract_integer_answer(answer) for answer in answers]
    
    summary_prompt = f'Make a summary answer for the question or pick the one you think is the correct one: "{question}" from all these given answers: 1. {answers[0]}, 2. {answers[1]}, 3. {answers[2]}, 4. {answers[3]} Please be very concise and to the point, just return the answer in the end as an integer in the format of Answer: xxx.'
    
    summary_response = together_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        messages=[{"role": "user", "content": summary_prompt}]
    )
    
    final_answer = extract_integer_answer(summary_response.choices[0].message.content)
    
    if final_answer is not None:
        log_result(question, answer, final_answer, output_filename, integer_answers)
        return int(final_answer) == int(answer)
    else:
        print("Failed to convert final response to an integer.")
        return False

def run_steps(question, answer):
    steps_response = together_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that can answer questions and provide step-by-step solutions."},
            {"role": "user", "content": f"Please provide the steps required to solve the following question: {question} without solving the question directly. Only provide the guidelines to solve the question."},
        ],
    )
    steps = steps_response.choices[0].message.content
    print(f"Steps generated: {steps}\n\n")
    
    result = asyncio.run(run_exo_cortex(f"Please provide the solution to the following question: {question}, given that the steps required to solve the question are: {steps}.", answer))
    return result

def evaluate_model(df, eval_function, output_filename):
    count = 0
    total = 0
    for index, row in df.iterrows():
        print(f"---------------------- Question: {index} -------------------------------")
        question = row['Question']
        answer = row['Answer']
        count += eval_function(question, answer)
        total += 1
    
    with open(output_filename, 'a') as f:
        print(f"Count correct: {count} of {total}", file=f)

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    df = pd.read_csv('AIME_Dataset_1983_2024.csv')
    aime_2024 = df[df['Year'].isin([2023, 2024])][['Question', 'Answer']][-2:]
    print(aime_2024)

    # Get command line arguments
    output_filename = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_FILENAME
    use_exo_cortex = sys.argv[2] == "exo_cortex" if len(sys.argv) > 2 else False

    # Run evaluation
    if use_exo_cortex:
        print("Using exo_cortex")
        evaluate_model(aime_2024, lambda q, a: run_steps(q, a), output_filename)
    else:
        print("Using gpt-4o")
        evaluate_model(aime_2024, lambda q, a: run_gpt4o(q, a, output_filename), output_filename)