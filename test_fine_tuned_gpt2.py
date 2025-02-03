import os
import unittest
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from math import exp
from datasets import Dataset
from rouge_score import rouge_scorer
import logging

logging.basicConfig(level=logging.INFO)

class TestFineTunedGPT2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset_path = "creative_texts.txt"
        cls.model_name = "gpt2"
        cls.fine_tuned_model_path = "./fine_tuned_gpt2"
        cls.tokenizer = GPT2Tokenizer.from_pretrained(cls.model_name)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        logging.info("\n[INFO] Setup completed. Model name and tokenizer initialized.\n")

    def load_model(self, model_path):
        logging.info(f"[INFO] Loading model from {model_path}...")
        return GPT2LMHeadModel.from_pretrained(model_path)

    def load_dataset(self, file_path, tokenizer, block_size=128):
        logging.info(f"[INFO] Loading dataset from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as file:
            texts = file.readlines()

        tokenized_texts = [
            tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=block_size,
                return_tensors="pt"
            )
            for text in texts
        ]
        dataset = Dataset.from_dict(
            {
                "input_ids": [t["input_ids"].squeeze() for t in tokenized_texts],
                "attention_mask": [t["attention_mask"].squeeze() for t in tokenized_texts],
            }
        )
        logging.info(f"[INFO] Dataset loaded with {len(dataset)} samples.")
        return dataset

    def test_dataset_loading(self):
        logging.info("\n[INFO] Running test: test_dataset_loading")
        dataset = self.load_dataset(self.dataset_path, self.tokenizer)
        self.assertGreater(len(dataset), 0, "Dataset is empty or failed to load.")
        logging.info("[INFO] Dataset loaded successfully.\n")

    def test_model_training(self):
        logging.info("\n[INFO] Running test: test_model_training")
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

        model = self.load_model(self.model_name)
        logging.info("[INFO] Model loaded for training.")

        dataset = Dataset.from_dict(
            {"input_ids": torch.randint(0, 50256, (10, 128)), "attention_mask": torch.ones((10, 128))}
        )
        logging.info("[INFO] Dummy dataset created for testing training pipeline.")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        training_args = TrainingArguments(
            output_dir="./test_results",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            logging_dir="./test_logs",
            logging_steps=5,
            save_steps=10,
            save_total_limit=1,
            overwrite_output_dir=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )

        try:
            logging.info("[INFO] Starting model training...")
            trainer.train()
            trainer.save_model("./test_fine_tuned_gpt2")
            logging.info("[INFO] Model training completed and saved.")
            self.assertTrue(os.path.exists("./test_fine_tuned_gpt2"), "Model not saved correctly.")
        except Exception as e:
            logging.error(f"[ERROR] Model training failed with error: {e}")
            self.fail(f"[ERROR] Model training failed with error: {e}")
    
    def test_text_generation(self):
        logging.info("\n[INFO] Running test: test_text_generation")
        model = self.load_model(self.fine_tuned_model_path)
        logging.info("[INFO] Fine-tuned model loaded for text generation.")
        
        input_text = "The sun dipped below the horizon , casting a warm glow over the ocean"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        attention_mask = torch.ones_like(input_ids)

        logging.info("[INFO] Generating text...")
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.9,
            top_k=40,
            top_p=0.7,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True
        )

        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        logging.info(f"[INFO] Generated text: {generated_text}")
        self.assertIn(input_text, generated_text, "[ERROR] Generated text does not contain the prompt.")
        logging.info("[INFO] Text generation test passed.\n")
    
if __name__ == "__main__":
    logging.info("\n[INFO] Starting test suite...")
    unittest.main()
