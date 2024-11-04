import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from PIL import Image
from IPython.display import display
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix

torch.manual_seed(42)
state = torch.load('AlexNetFruitImageCNNModel.pth')

# Define model.

model = models.alexnet(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Linear(9216, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 3),
    nn.LogSoftmax(dim=1),
)

# Define loss function.
criterion = nn.CrossEntropyLoss()

# Define optimizer.
optimizer = torch.optim.Adagrad(model.classifier.parameters(), lr=0.001)


# Load data.
model.load_state_dict(state['model_state_dict'])
criterion.load_state_dict(state['criterion_state_dict'])
optimizer.load_state_dict(state['optimizer_state_dict'])

model.eval()
criterion.eval()

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


test_data = datasets.ImageFolder('testdata', transform=test_transform)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

tst_corr = 0
# test_losses = []
test_correct = []
with torch.no_grad():
    for (X_test, y_test) in test_loader:
        # Apply the model
        y_val = model(X_test)
        # Tally the number of correct predictions
        predicted = torch.max(y_val.data, 1)[1] 
        tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    # test_losses.append(loss)
    test_correct.append(tst_corr)

def get_test_accuracies(test_correct):
    test_accuracies = []
    for n in range(len(test_correct)):
        test_accuracy = test_correct[n]*100/len(test_data)
        test_accuracies.append(test_accuracy)
    return test_accuracies

test_accuracies = get_test_accuracies(test_correct)

print(test_correct)
print(f'Test accuracy: {test_accuracies[-1].item():.3f}%')
