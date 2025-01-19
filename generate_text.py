from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('./fine_tuned_gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tokenizer.pad_token = tokenizer.eos_token  

input_text = "Once Upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

attention_mask = (input_ids != tokenizer.pad_token_id).type(input_ids.dtype)

output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, attention_mask=attention_mask)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
