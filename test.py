import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from huggingface_hub import login
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from concurrent.futures import as_completed
input_file = "/home/cs601-zmou1/Math-Reasoning-on-Frequency/datasets/arithmetic_dataset_onedigit.csv"
data = pd.read_csv(input_file)
answers = {}

import torch
model_name = "meta-llama/Llama-2-7b-chat-hf"
login("**")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
)

def solve_math_problems(questions):
    with torch.no_grad():
        sequences = pipeline(
            [f'Provide only the numerical answer for this math problem: {question} = ？' for question in questions],
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=200,
            truncation=True
        )
        # Extract only the content after "Answer:" using regex
        answers = [
            re.search(r"Answer:\s*(.*)", sequence[0]["generated_text"]).group(1).strip()
            if re.search(r"Answer:\s*(.*)", sequence[0]["generated_text"])
            else ""
            for sequence in sequences
        ]
    return answers

def process_column(column_name):
    return solve_math_problems(data[column_name].dropna())

# Run each column in parallel using ThreadPoolExecutor and track progress
with ThreadPoolExecutor(max_workers=4) as executor:
    # Wrap the futures in a tqdm progress bar
    futures = {executor.submit(process_column, column): column for column in ['Question Symbolic', 'Question Text 1', 'Question Text 2', 'Question Text 3']}

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing columns"):
        column_name = futures[future]
        try:
            answers[column_name] = future.result()
        except Exception as e:
            print(f"Error processing column {column_name}: {e}")

# Create a new DataFrame with the results
output_df = pd.DataFrame({
    'Answer Symbolic': answers['Question Symbolic'],
    'Answer Text 1': answers['Question Text 1'],
    'Answer Text 2': answers['Question Text 2'],
    'Answer Text 3': answers['Question Text 3']
})

# Save the output to a new CSV file
output_df.to_csv("generated_answers_onedigit.csv", index=False)