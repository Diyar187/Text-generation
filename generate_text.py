from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Laden des feinabgestimmten Modells
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Sicherstellen, dass der Pad-Token korrekt gesetzt ist
tokenizer.pad_token = tokenizer.eos_token  # Setzt den Pad-Token als den EOS-Token

# Eingabetext definieren
input_text = "Once Upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Erstellen der Attention Mask
attention_mask = (input_ids != tokenizer.pad_token_id).type(input_ids.dtype)

# Textgenerierung
output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, attention_mask=attention_mask)

# Ausgabe dekodieren und drucken
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
