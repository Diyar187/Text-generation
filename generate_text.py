from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained('./fine_tuned_gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tokenizer.pad_token = tokenizer.eos_token

input_text = "Once Upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

attention_mask = torch.ones_like(input_ids)

def generate_text(input_text, max_length=200, temperature=0.7, top_k=50, top_p=0.9):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    
    def ensure_sentence_completion(text):
        sentence_end = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if sentence_end != -1:
            return text[: sentence_end + 1]  
        return text  
    return ensure_sentence_completion(generated_text)

generated_text = generate_text(input_text)
print(generated_text)
