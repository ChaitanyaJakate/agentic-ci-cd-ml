# Testing GitHub Actions CI/CD

from dataset_31_march_2025 import model1, train_loader, test_loader, torch, nn
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model1.parameters())
for epoch in range(2):
    for X, y in train_loader:
        out = model1(X)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} completed.")
