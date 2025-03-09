## Fine-Tuning GPT-2 für kreatives Schreiben

Dieses Projekt befasst sich mit dem Finetuning eines GPT-2-Modells zur Generierung kreativer Texte. Es umfasst das Training des Modells mit einer benutzerdefinierten Datensammlung, die Generierung neuer Texte sowie Unit-Tests zur Sicherstellung der Funktionalität.


## Verzeichnisstruktur
```
📁 project_root
├── 📝 fine_tune_gpt2.py          # Skript zum Trainieren von GPT-2
├── 📝 generate_text.py           # Skript zur Textgenerierung
├── 🧪 test_fine_tuned_gpt2.py    # Unittests für das trainierte Modell
├── 📄 creative_texts.txt         # Trainingsdaten für das Modell
├── 📖 README.md                  # Projektdokumentation
```

## Nutzung
### 1️⃣ **Finetuning des Modells**
```bash
python fine_tune_gpt2.py
```
Das Modell wird mit den in `creative_texts.txt` gespeicherten Beispielen trainiert und im Verzeichnis `./fine_tuned_gpt2` gespeichert.

### 2️⃣ **Generierung von Texten**
```bash
python generate_text.py
```
Das Skript lädt das trainierte Modell und generiert Texte basierend auf einer vorgegebenen Eingabe.

### 3️⃣ **Tests ausführen**
```bash
python -m unittest test_fine_tuned_gpt2.py
```
Die Tests überprüfen, ob das Modell korrekt trainiert wurde und Texte sinnvoll generiert.

## Anpassungen
- Die Trainingsdaten in `creative_texts.txt` können erweitert oder angepasst werden.
- Die Hyperparameter für das Training und die Textgenerierung (z.B. `temperature`, `top_k`, `top_p`) können in den jeweiligen Skripten modifiziert werden.



