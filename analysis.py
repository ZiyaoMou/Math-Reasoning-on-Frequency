from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False)
engine = InfiniGramEngine(index_dir='index/v4_pileval_llama', eos_token_id=tokenizer.eos_token_id)

input_ids = tokenizer.encode('natural language processing')
engine.count(input_ids=input_ids)