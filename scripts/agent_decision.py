import os
os.environ["USE_TF"] = "0"  # Prevent importing TensorFlow/Keras

from transformers import pipeline

# Force PyTorch backend
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    framework="pt"
)

sequence = "Trigger CI/CD workflow by editing run_pytorch.py"
labels = ["pytorch", "svm", "ann"]

result = classifier(sequence, labels)
print(result)

