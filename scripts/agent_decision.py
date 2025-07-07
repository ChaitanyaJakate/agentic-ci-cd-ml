from transformers import pipeline

# Force PyTorch framework to avoid Keras/TensorFlow
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    framework="pt"
)

sequence = "Trigger CI/CD workflow by editing run_pytorch.py"
labels = ["pytorch", "svm", "ann"]

result = classifier(sequence, labels)
print(result)

