from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# Pfad zu deinem Datensatz
dataset_path = "creative_texts.txt"  # Stelle sicher, dass der Pfad korrekt ist

# Lade Tokenizer und Modell
model_name = "gpt2"  # Nutze ein kleineres Modell, z. B. "gpt2" oder "gpt2-small"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Setze den Padding-Token auf den EOS-Token
tokenizer.pad_token = tokenizer.eos_token

# Lade den Datensatz
def load_dataset(file_path, tokenizer, block_size=128):
    """
    Lade einen Textdatensatz und bereite ihn für das Training vor.
    
    Args:
    - file_path: Pfad zur Textdatei.
    - tokenizer: Tokenizer für die Textverarbeitung.
    - block_size: Länge der Textblöcke für das Training.

    Returns:
    - Dataset: Der geladene Datensatz.
    """
    try:
        dataset = Dataset.from_text(file_path)
        print(f"Datensatz erfolgreich geladen. Anzahl der Zeilen: {len(dataset)}")
    except Exception as e:
        print(f"Fehler beim Laden des Datensatzes: {e}")
        return None
    
    # Tokenisiere den Text
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=block_size)
    
    # Tokenisiere alle Beispiele im Datensatz
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Debug-Ausgabe: Überprüfe, ob die Tokenisierung erfolgreich war
    print("Tokenisierte Beispiele:")
    for i in range(min(5, len(tokenized_datasets))):
        print(tokenizer.decode(tokenized_datasets[i]["input_ids"], skip_special_tokens=True))
    
    return tokenized_datasets

# Lade den Datensatz
train_dataset = load_dataset(dataset_path, tokenizer)

# Überprüfe, ob der Datensatz geladen wurde
if train_dataset is None or len(train_dataset) == 0:
    print("Fehler: Der Datensatz konnte nicht geladen werden oder ist leer!")
else:
    # Überprüfe, wie viele Beispiele im Datensatz enthalten sind
    print(f"Anzahl der Trainingsbeispiele: {len(train_dataset)}")

# Daten-Handler für das Sprachmodell
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Trainingsargumente
training_args = TrainingArguments(
    output_dir="./results",        # Verzeichnis für die gespeicherten Ergebnisse
    overwrite_output_dir=True,    # Vorhandene Dateien überschreiben
    num_train_epochs=3,           # Anzahl der Trainingsepochen
    per_device_train_batch_size=2, # Batch-Größe
    save_steps=10,                # Anzahl der Schritte zwischen Speichervorgängen
    logging_dir="./logs",         # Verzeichnis für Log-Dateien
    logging_steps=5               # Schritte zwischen Logs
)

# Trainer-Objekt initialisieren
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

# Training starten, nur wenn der Datensatz korrekt geladen wurde
if train_dataset is not None and len(train_dataset) > 0:
    trainer.train()

    # Feinjustiertes Modell speichern
    output_dir = "./fine_tuned_gpt2"
    trainer.save_model(output_dir)
    print(f"Modell erfolgreich gespeichert in: {output_dir}")
else:
    print("Das Training wurde nicht gestartet, da der Datensatz leer oder fehlerhaft ist.")
