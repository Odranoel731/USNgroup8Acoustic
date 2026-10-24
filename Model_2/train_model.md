Loading the data:

X = np.load(f"{FEATURES}/X_train.npy")
y = np.load(f"{FEATURES}/y_train.npy")
X_test = np.load(f"{FEATURES}/X_test.npy")
y_test = np.load(f"{FEATURES}/y_test.npy")
This loads the spectrogram images and their labels:

X contains the engine-sound spectrograms.

y contains the correct class: 0 = good, 1 = broken, 2 = heavy load.

X_test and y_test are kept separate so the model is tested on sounds it did not train on.
X = X[:, None, :, :].astype("float32")
This adds a channel dimension so each spectrogram is treated like a one-channel image, similar to a black-and-white image.

Splitting the data

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
The training data is split into:

80% training data: used to teach the model.

20% validation data: used after each epoch to check whether the model is improving on data it has not trained on.

stratify=y keeps the same proportion of good, broken, and heavy load in both sets.

random_state=42 makes the split repeatable.

Batches

train_loader = DataLoader(..., batch_size=16, shuffle=True)
Instead of showing the model all training samples at once, it shows 16 spectrograms at a time. After each batch, the model updates itself a little. shuffle=True changes the order each epoch so the model does not simply learn the order of the files.

The AI model

class EngineCNN(nn.Module):
This is the neural network. It looks at each spectrogram and learns patterns that separate the three engine conditions. It has convolution layers to find sound patterns, pooling layers to reduce the image size, and final linear layers to choose one of the three classes.


device = "cuda" if torch.cuda.is_available() else "cpu"
This uses the GPU if available; otherwise it uses the CPU.


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion measures how wrong the model’s predictions are.

optimizer adjusts the model’s internal weights to make those mistakes smaller.

lr=0.001 is the learning rate: how large each adjustment is.

What one epoch does

for epoch in range(MAX_EPOCHS):
Each loop is one epoch: one complete pass through all training samples.


model.train()
This puts the model in training mode, enabling learning-related behavior such as dropout.


for xb, yb in train_loader:
This goes through the training data in batches of 16. xb is a batch of spectrograms, and yb contains the correct labels for those spectrograms.


optimizer.zero_grad()
loss = criterion(model(xb), yb)
loss.backward()
optimizer.step()
This is the core learning step:

model(xb) makes predictions for the 16 spectrograms.

criterion(...) calculates how wrong those predictions are.

loss.backward() calculates which weights caused the error.

optimizer.step() slightly changes those weights to improve future predictions.

So, during one epoch, the model sees every training batch once and updates itself after each batch.

Checking after each epoch

model.eval()
This switches to evaluation mode for validation.


with torch.no_grad():
This tells PyTorch not to update the model while checking validation data. Validation is only for measuring performance, not for learning.


val_loss /= len(val_loader.dataset)
This calculates the average validation loss over all validation samples. Lower validation loss usually means better generalization.

Early stopping

if val_loss < best_val_loss - MIN_IMPROVEMENT:
If validation loss improves enough, the model’s current weights are saved:


torch.save(model.state_dict(), "engine_cnn.pth")
Otherwise:


patience_counter += 1
If validation loss does not improve for PATIENCE = 7 epochs in a row, training stops early. That is why your run said:

text
Early stopping after 10 epochs.
It means the model had its best validation loss around epoch 3, then did not improve meaningfully for the next 7 epochs.

Final test

model.load_state_dict(torch.load("engine_cnn.pth"))
This loads the best saved version of the model, not necessarily the final epoch’s version.


predictions.extend(model(xb.to(device)).argmax(1).cpu().numpy())
For each test spectrogram, the model gives three scores. argmax(1) chooses the class with the highest score: good, broken, or heavy load.


print(classification_report(...))
print(confusion_matrix(...))
Finally, the code compares its predictions with the true test labels. The classification report shows precision, recall, and F1-score; the confusion matrix shows exactly which classes were confused with each other