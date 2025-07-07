import os
os.environ["USE_TF"] = "0"  # Prevent Hugging Face from touching TensorFlow

from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    framework="pt"
)

sequence = "Trigger CI/CD workflow by editing run_pytorch.py"
labels = ["pytorch", "svm", "ann"]

result = classifier(sequence, labels)
print(result)

