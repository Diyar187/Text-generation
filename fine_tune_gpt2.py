import logging
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

logging.basicConfig(level=logging.INFO)

dataset_path = "creative_texts.txt"

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

def load_dataset(file_path, tokenizer, block_size=128):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            texts = file.readlines()
    except Exception as e:
        logging.error(f"Error loading the dataset: {e}")
        return None

    
    tokenized_texts = []
    for text in texts:
        try:
            tokenized = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=block_size,
                return_tensors="pt"
            )
            tokenized_texts.append(
                {"input_ids": tokenized["input_ids"].squeeze(), "attention_mask": tokenized["attention_mask"].squeeze()}
            )
        except Exception as e:
            logging.error(f"Error tokenizing the text: {text}")
            continue

    dataset = Dataset.from_dict(
        {
            "input_ids": [t["input_ids"] for t in tokenized_texts],
            "attention_mask": [t["attention_mask"] for t in tokenized_texts],
        }
    )
    logging.info(f"Dataset successfully loaded. Number of rows: {len(dataset)}")
    return dataset

train_dataset = load_dataset(dataset_path, tokenizer)

if train_dataset is None or len(train_dataset) == 0:
    logging.error("Error: The dataset could not be loaded or is empty!")
    exit()

logging.info(f"Number of training examples: {len(train_dataset)}")

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
    logging_steps=5,
    save_total_limit=2,  
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()
output_dir = "./fine_tuned_gpt2"
trainer.save_model(output_dir)
logging.info(f"Model successfully saved in: {output_dir}")
