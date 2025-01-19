from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

dataset_path = "creative_texts.txt"  

model_name = "gpt2" 
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

def load_dataset(file_path, tokenizer, block_size=128):
    try:
        dataset = Dataset.from_text(file_path)
        print(f"Data set loaded successfully. Number of lines: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=block_size)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    print("Tokenized examples:")
    for i in range(min(5, len(tokenized_datasets))):
        print(tokenizer.decode(tokenized_datasets[i]["input_ids"], skip_special_tokens=True))
    
    return tokenized_datasets

train_dataset = load_dataset(dataset_path, tokenizer)

if train_dataset is None or len(train_dataset) == 0:
    print("Error: The data set could not be loaded or is empty")
else:
    print(f"Number of training examples: {len(train_dataset)}")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir="./results",        
    overwrite_output_dir=True,    
    num_train_epochs=3,          
    per_device_train_batch_size=2, 
    save_steps=10,                
    logging_dir="./logs",       
    logging_steps=5              
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

if train_dataset is not None and len(train_dataset) > 0:
    trainer.train()


    output_dir = "./fine_tuned_gpt2"
    trainer.save_model(output_dir)
    print(f"Model successfully saved in: {output_dir}")
else:
    print("The training did not start because the data set is empty or incorrect.")
