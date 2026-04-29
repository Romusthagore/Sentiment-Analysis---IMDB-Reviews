# 🎭 Sentiment Analysis on IMDB Reviews

> Fine-tuning DistilBERT for binary sentiment classification (Positive/Negative)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Romusthagore/Sentiment-Analysis---IMDB-Reviews) [![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/) [![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)

## 🏆 Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 90.0% |
| **Precision** | 86.6% |
| **Recall** | 93.9% |
| **F1 Score** | 90.1% |

## 🎯 What it does

Analyzes movie reviews and classifies them as **Positive** or **Negative**:

| Review | Prediction | Confidence |
|--------|------------|------------|
| "This movie was absolutely amazing!" | ✅ Positive | 99.2% |
| "Terrible film, waste of time." | ❌ Negative | 98.7% |
| "It was okay, nothing special." | 😐 Negative | 62.2% |
| "Best movie I've seen this year!" | ✅ Positive | 98.5% |

## 🏗️ Model Architecture

| Component | Description |
|-----------|-------------|
| **Base Model** | DistilBERT-base-uncased |
| **Task** | Sequence Classification (2 classes) |
| **Framework** | Hugging Face Transformers + PyTorch |
| **Dataset** | IMDB (25k train / 25k test) |

## 🚀 Quick Start

### Option 1: One-click (easiest) ☁️

Click the **"Open in GitHub Codespaces"** badge above — an environment will launch automatically with everything installed.

### Option 2: Local installation 💻

```bash
# Clone the repository
git clone https://github.com/Romusthagore/Sentiment-Analysis---IMDB-Reviews.git
cd Sentiment-Analysis---IMDB-Reviews

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook Sentiment_Analysis_on.ipynb


Option 3: Use the trained model directly 🚀
python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model (once pushed to Hugging Face Hub)
model = AutoModelForSequenceClassification.from_pretrained("Romusthagore/imdb-sentiment-model")
tokenizer = AutoTokenizer.from_pretrained("Romusthagore/imdb-sentiment-model")

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    sentiment = "Positive" if probs[0][1] > 0.5 else "Negative"
    confidence = probs[0][1].item() if sentiment == "Positive" else 1 - probs[0][1].item()
    return sentiment, confidence

# Test it!
print(predict_sentiment("I loved this movie!"))  # ('Positive', 0.99)
📊 Training Details
Hyperparameter	Value
Training samples	5,000
Evaluation samples	1,000
Epochs	3
Batch size (train/eval)	16 / 32
Learning rate	2e-5
Max sequence length	512
Optimizer	AdamW
Hardware	GPU (CUDA)
📈 Training Curves
(Add your loss/accuracy plots here)

text
Training Loss:     0.110 → Validation Loss: 0.340
Accuracy:          0.903 → F1 Score:        0.902
📁 Repository Structure
text
Sentiment-Analysis---IMDB-Reviews/
├── Sentiment_Analysis_on.ipynb   # Complete training notebook
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
└── README.md                      # This file
📝 Dependencies
text
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
scikit-learn>=1.2.0
numpy>=1.24.0
🔮 Future Improvements
Push model to Hugging Face Hub

Add live demo with Gradio/Streamlit

Support for multi-language reviews

Fine-tune on other datasets (Twitter, Amazon reviews)

📄 License
This project is licensed under the MIT License — see the LICENSE file for details.

👤 Author
Romuald AHOMAGNON
https://img.shields.io/badge/GitHub-Romusthagore-black
https://img.shields.io/badge/LinkedIn-Romuald-blue
