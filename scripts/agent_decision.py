
import sys
from transformers import pipeline

commit_message = sys.argv[1]
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
labels = ["svm", "ann", "pytorch"]
result = classifier(commit_message, labels)
decision = result['labels'][0]
print(f"Decision: {decision}")
with open("model_type.txt", "w") as f:
    f.write(decision)
