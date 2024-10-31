from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch

model_name = "meta-llama/Llama-2-7b-chat-hf"  # Or "meta-llama/Llama-2-13b-chat" for larger model

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

def solve_math_problem(question):
    with torch.no_grad():
        sequences = pipeline(
            f'Solve this math problem: {question}',
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=200,
        )
        answer = sequences[0]["generated_text"]
    return answer

# Test the function
question = "What is 3 + 4?"
print(solve_math_problem(question))