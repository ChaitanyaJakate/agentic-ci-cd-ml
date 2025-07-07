
from dataset_31_march_2025 import svm, new_test_ss, y_test
from sklearn.metrics import accuracy_score
y_pred = svm.predict(new_test_ss)
acc = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {acc}")
