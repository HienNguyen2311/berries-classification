#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
import random
from PIL import Image
from IPython.display import display
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.image import imread
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix


# # Load the data

# In[2]:


root = os.getcwd()
path = os.path.join(root, 'traindata')


# In[3]:


def get_image_names(path):
    img_names = []

    for folder, subfolders, filenames in os.walk(path):
        for img in filenames:
            img_names.append(folder+'\\'+img)

    print('Number of images: ',len(img_names))
    return img_names


# In[4]:


img_names = get_image_names(path)


# In[5]:


img_sizes = []
rejected = []

for item in img_names:
    try:
        with Image.open(item) as img:
            img_sizes.append(img.size)
    except:
        rejected.append(item)
        
print(f'Images:  {len(img_sizes)}')
print(f'Rejects: {len(rejected)}')


# # EDA

# In[6]:


df = pd.DataFrame(img_sizes)
# Run summary statistics on image widths
df_w = pd.DataFrame(df[0].describe())
df_w.round(1)


# In[7]:


# Run summary statistics on image heights
df_h = pd.DataFrame(df[1].describe())
df_h.round(1)


# In[8]:


randomlist = []
randomlist = random.sample(range(0, 3599), 20)
print(randomlist)


# In[9]:


for i in randomlist:
    fruit = Image.open(img_names[i])
    display(fruit)


# In[10]:


def plotHist(img):
  plt.figure(figsize=(10,5))
  plt.subplot(1,2,1)
  plt.imshow(img, cmap='gray')
  plt.axis('off')
  histo = plt.subplot(1,2,2)
  histo.set_ylabel('Count')
  histo.set_xlabel('Pixel Intensity')
  plt.hist(img.flatten(), bins=10, lw=0, color='r', alpha=0.5)


# In[11]:


img = mpimg.imread(img_names[300])
plotHist(img)


# In[12]:


cherry_path = os.path.join(path, 'cherry')
strawberry_path = os.path.join(path, 'strawberry')
tomato_path = os.path.join(path, 'tomato')


# In[13]:


cherry_img_names = get_image_names(cherry_path)


# In[14]:


strawberry_img_names = get_image_names(strawberry_path)


# In[15]:


tomato_img_names = get_image_names(tomato_path)


# In[16]:


def get_fruit_rgb(fruit_path):
    r_fruits = []
    g_fruits = []
    b_fruits = []
    for i in range(len(fruit_path)):
        fruit = Image.open(fruit_path[i])
        rgb_image = fruit.convert('RGB')
        r, g, b = rgb_image.getpixel((0, 0))
        r_fruits.append(r)
        g_fruits.append(g)
        b_fruits.append(b)
    return r_fruits, g_fruits, b_fruits


# In[17]:


r_cherry, g_cherry, b_cherry = get_fruit_rgb(cherry_img_names)
r_strawberry, g_strawberry, b_strawberry = get_fruit_rgb(strawberry_img_names)
r_tomato, g_tomato, b_tomato = get_fruit_rgb(tomato_img_names)


# In[18]:


rgb_cherry = pd.DataFrame(list(zip(r_cherry, g_cherry, b_cherry)),
               columns = ['R', 'G', 'B'])
rgb_cherry = rgb_cherry.assign(Fruit = 'cherry')


# In[19]:


rgb_strawberry = pd.DataFrame(list(zip(r_strawberry, g_strawberry, b_strawberry)),
               columns = ['R', 'G', 'B'])
rgb_strawberry = rgb_strawberry.assign(Fruit = 'strawberry')


# In[20]:


rgb_tomato = pd.DataFrame(list(zip(r_tomato, g_tomato, b_tomato)),
               columns = ['R', 'G', 'B'])
rgb_tomato = rgb_tomato.assign(Fruit = 'tomato')


# In[21]:


fruit_rgb = rgb_cherry.append(rgb_strawberry, ignore_index=True)
fruit_rgb = fruit_rgb.append(rgb_tomato, ignore_index=True)


# In[22]:


fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
sns.boxplot(x='Fruit', y='R', data=fruit_rgb, ax=axes[0])
sns.boxplot(x='Fruit', y='G', data=fruit_rgb, ax=axes[1])
sns.boxplot(x='Fruit',y='B', data=fruit_rgb, ax=axes[2])
fig.tight_layout()
plt.show()


# # Apply data pre-processing/ data transformation

# In[23]:


train_transform = transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


# In[24]:


train_data = datasets.ImageFolder(os.path.join(root, 'traindata'), transform=train_transform)
test_data = datasets.ImageFolder(os.path.join(root, 'testdata'), transform=test_transform)

torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100, shuffle=True)

class_names = train_data.classes

print(class_names)
print(f'Training images available: {len(train_data)}')
print(f'Testing images available:  {len(test_data)}')


# In[25]:


# Grab the first batch of 100 images
for images,labels in train_loader: 
    break

# Print the labels
print('Label:', labels.numpy())
print('Class:', *np.array([class_names[i] for i in labels]))

im = make_grid(images)

# Inverse normalize the images
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
im_inv = inv_normalize(im)

# Print the images
plt.figure(figsize=(80,40))
plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)));


# # MLP baseline model

# In[26]:


class MultilayerPerceptron(nn.Module):
    def __init__(self, in_sz=150528, out_sz=3, layers=[120,84]):
        super().__init__()
        self.fc1 = nn.Linear(in_sz, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], out_sz)
    
    def forward(self,X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)


# In[27]:


torch.manual_seed(42)
model_mlp = MultilayerPerceptron()
model_mlp


# In[28]:


def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>8}')
    print(f'______\n{sum(params):>8}')


# In[29]:


count_parameters(model_mlp)


# ## Train the model

# In[30]:


criterion_mlp = nn.CrossEntropyLoss()
optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=0.001, weight_decay=1e-5)
epochs_mlp = 8


# In[31]:


for images, labels in train_loader:
    print('Batch shape:', images.size())
    break


# In[32]:


images.view(100,-1).size()


# In[33]:


def train_model(model, train_loader, test_loader, criterion, optimizer, epochs):
    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []
    for i in range(epochs):
        trn_corr = 0
        tst_corr = 0
        # Run the training batches
        for (X_train, y_train) in train_loader:
            # Apply the model
            if model == model_mlp:
                y_pred = model(X_train.view(100, -1))
            else:
                y_pred = model(X_train)
            loss = criterion(y_pred, y_train) 
            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr        
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Update train loss & accuracy for the epoch 
        train_losses.append(loss)
        train_correct.append(trn_corr)
        # Run the testing batches
        with torch.no_grad():
            for (X_test, y_test) in test_loader:
                # Apply the model
                if model == model_mlp:
                    y_val = model(X_test.view(100, -1))
                else:
                    y_val = model(X_test)
                # Tally the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1] 
                tst_corr += (predicted == y_test).sum()
        loss = criterion(y_val, y_test)
        test_losses.append(loss)
        test_correct.append(tst_corr)
    return train_losses, test_losses, train_correct, test_correct


# In[34]:


def get_train_accuracies(train_correct):
    train_accuracies = []
    for n in range(len(train_correct)):
        train_accuracy = train_correct[n]*100/len(train_data)
        train_accuracies.append(train_accuracy)
    return train_accuracies


# In[35]:


def get_test_accuracies(test_correct):
    test_accuracies = []
    for n in range(len(test_correct)):
        test_accuracy = test_correct[n]*100/len(test_data)
        test_accuracies.append(test_accuracy)
    return test_accuracies


# In[36]:


def plot_losses(train_losses, test_losses):
    plt.plot(train_losses, label='training loss')
    plt.plot(test_losses, label='validation loss')
    plt.xticks(np.arange(len(train_losses)), np.arange(1, len(train_losses)+1))
    plt.title('Loss at each epoch')
    plt.legend()


# In[37]:


def plot_accuracies(train_accuracies, test_accuracies):
    plt.plot(train_accuracies, label='training accuracy')
    plt.plot(test_accuracies, label='validation accuracy')
    plt.xticks(np.arange(len(train_accuracies)), np.arange(1, len(train_accuracies)+1))
    plt.title('Accuracy at each epoch')
    plt.legend()


# In[38]:


train_losses_mlp, test_losses_mlp, train_correct_mlp, test_correct_mlp = train_model(model_mlp, train_loader, test_loader, criterion_mlp, optimizer_mlp, epochs_mlp)


# In[39]:


train_accuracies_mlp = get_train_accuracies(train_correct_mlp)
test_accuracies_mlp = get_test_accuracies(test_correct_mlp)


# In[40]:


plot_losses(train_losses_mlp, test_losses_mlp)


# In[41]:


plot_accuracies(train_accuracies_mlp, test_accuracies_mlp)


# ## Evaluate model performance

# In[42]:


print(f'Train accuracy: {train_accuracies_mlp[-1].item():.2f}%')
print(f'Test accuracy: {test_accuracies_mlp[-1].item():.2f}%')
print(f'Train loss: {train_losses_mlp[-1].item():.2f}')
print(f'Test loss: {test_losses_mlp[-1].item():.2f}')


# ## Display the confusion matrix

# In[43]:


def create_cf(test_data, model):
    # Create a loader for the entire the test set
    test_load_all = DataLoader(test_data, batch_size=900, shuffle=True)
    with torch.no_grad():
        correct = 0
        for X_test, y_test in test_load_all:
            if model == model_mlp:
                y_val = model_mlp(X_test.view(900, -1)) 
            else:
                y_val = model(X_test) 
            predicted = torch.max(y_val,1)[1]
            correct += (predicted == y_test).sum()

    arr = confusion_matrix(y_test.view(-1), predicted.view(-1))
    df_cm = pd.DataFrame(arr, class_names, class_names)
    plt.figure(figsize = (9,6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("Predicted value")
    plt.ylabel("True value")
    plt.show()
    
    return predicted, X_test, y_test


# In[44]:


predicted_mlp, X_test_mlp, y_test_mlp = create_cf(test_data, model_mlp)


# ## Examine the misses

# In[45]:


def get_missed_predictions(predicted, X_test, y_test):
    misses = np.array([])
    for i in range(len(predicted.view(-1))):
        if predicted[i] != y_test[i]:
            misses = np.append(misses,i).astype('int64')
    # Set up an iterator to feed batched rows
    r = 20   # row size
    row = iter(np.array_split(misses,len(misses)//r+1))
    np.set_printoptions(formatter=dict(int=lambda x: f'{x:5}')) # to widen the printed array

    nextrow = next(row)
    lbls = y_test.index_select(0,torch.tensor(nextrow)).numpy()
    gues = predicted.index_select(0,torch.tensor(nextrow)).numpy()
    print("Index:", nextrow)
    print("Label:", lbls)
    print("Class: ", *np.array([class_names[i] for i in lbls]))
    print()
    print("Guess:", gues)
    print("Class: ", *np.array([class_names[i] for i in gues]))

    images = X_test.index_select(0,torch.tensor(nextrow))
    im = make_grid(images, nrow=r)
    inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225])
    im_inv = inv_normalize(im)
    plt.figure(figsize=(50,25))
    plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)))


# In[46]:


get_missed_predictions(predicted_mlp, X_test_mlp, y_test_mlp)


# # CNN model

# In[47]:


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1) #3 input color channels, 6 filters (output channels), kernel size 3x3, stride 1
        self.conv2 = nn.Conv2d(6, 16, 3, 1) #6 filters, 16 filters
        self.fc1 = nn.Linear(54*54*16, 120) #120 flattened out neurons
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
        self.dropout = nn.Dropout(0.25)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2) #kernel size 2, stride 2
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54*54*16)
        X = self.dropout(X)
        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)


# In[48]:


torch.manual_seed(42)
CNNmodel = ConvolutionalNetwork()
CNNmodel


# In[49]:


count_parameters(CNNmodel)


# ## Train the model

# In[50]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.001)
epochs = 8


# In[51]:


train_losses, test_losses, train_correct, test_correct = train_model(CNNmodel, train_loader, test_loader, criterion, optimizer, epochs)


# In[52]:


train_accuracies = get_train_accuracies(train_correct)
test_accuracies = get_test_accuracies(test_correct)


# In[53]:


plot_losses(train_losses, test_losses)


# In[54]:


plot_accuracies(train_accuracies, test_accuracies)


# ## Evaluate model performance

# In[55]:


print(f'Train accuracy: {train_accuracies[-1].item():.2f}%')
print(f'Test accuracy: {test_accuracies[-1].item():.2f}%')
print(f'Train loss: {train_losses[-1].item():.2f}')
print(f'Test loss: {test_losses[-1].item():.2f}')


# ## Display the confusion matrix

# In[56]:


predicted, X_test, y_test = create_cf(test_data, CNNmodel)


# ## Examine the misses

# In[57]:


get_missed_predictions(predicted, X_test, y_test)


# # Investigate loss function

# ## Negative Log-Likelihood Loss Function

# In[58]:


criterion_nll = nn.NLLLoss()
optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.001)
epochs = 8


# In[59]:


train_losses_nll, test_losses_nll, train_correct_nll, test_correct_nll = train_model(CNNmodel, train_loader, test_loader, criterion_nll, optimizer, epochs)


# In[60]:


train_accuracies_nll = get_train_accuracies(train_correct_nll)
test_accuracies_nll = get_test_accuracies(test_correct_nll)


# In[61]:


def plot_investigation_losses(train_losses_investigation, invest_train_label, test_losses_investigation, invest_test_label):
    plt.figure(figsize = (10,6))
    plt.plot(train_losses, label='Initial CNNmodel training loss')
    plt.plot(train_losses_investigation, label=invest_train_label)
    plt.plot(test_losses, label='Initial CNNmodel validation loss')
    plt.plot(test_losses_investigation, label=invest_test_label)
    plt.xticks(np.arange(len(train_losses)), np.arange(1, len(train_losses)+1))
    plt.title('Loss at each epoch')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# In[62]:


plot_investigation_losses(train_losses_nll, 'CNNmodel with NLL training loss', test_losses_nll, 'CNNmodel with NLL validation loss')


# In[63]:


def plot_investigation_accuracies(train_accuracies_investigation, invest_train_label, test_accuracies_investigation, invest_test_label):
    plt.figure(figsize = (10,6))
    plt.plot(train_accuracies, label='Initial CNNmodel training accuracy')
    plt.plot(train_accuracies_investigation, label=invest_train_label)
    plt.plot(test_accuracies, label='Initial CNNmodel validation accuracy')
    plt.plot(test_accuracies_investigation, label=invest_test_label)
    plt.xticks(np.arange(len(train_accuracies)), np.arange(1, len(train_accuracies)+1))
    plt.title('Accuracy at each epoch')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# In[64]:


plot_investigation_accuracies(train_accuracies_nll, 'CNNmodel with NLL training accuracy', test_accuracies_nll, 'CNNmodel with NLL validation accuracy')


# In[65]:


print(f'Train accuracy: {train_accuracies_nll[-1].item():.2f}%')
print(f'Test accuracy: {test_accuracies_nll[-1].item():.2f}%')
print(f'Train loss: {train_losses_nll[-1].item():.2f}')
print(f'Test loss: {test_losses_nll[-1].item():.2f}')


# # Investigate minibatch size 

# ## Batch size = 32

# In[66]:


def load_dataset(train_data, test_data, batch_size):
    torch.manual_seed(42)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


# In[67]:


train_loader1, test_loader1 = load_dataset(train_data, test_data, 32)


# In[68]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.001)
epochs = 8


# In[69]:


train_losses_32, test_losses_32, train_correct_32, test_correct_32 = train_model(CNNmodel, train_loader1, test_loader1, criterion, optimizer, epochs)


# In[70]:


train_accuracies_32 = get_train_accuracies(train_correct_32)
test_accuracies_32 = get_test_accuracies(test_correct_32)


# In[71]:


plot_investigation_losses(train_losses_32, 'CNNmodel with batch size of 32 - training loss', test_losses_32, 'CNNmodel with batch size of 32 - validation loss')


# In[72]:


plot_investigation_accuracies(train_accuracies_32, 'CNNmodel with batch size of 32 - training accuracy', test_accuracies_32, 'CNNmodel with batch size of 32 - validation accuracy')


# In[73]:


print(f'Train accuracy: {train_accuracies_32[-1].item():.2f}%')
print(f'Test accuracy: {test_accuracies_32[-1].item():.2f}%')
print(f'Train loss: {train_losses_32[-1].item():.2f}')
print(f'Test loss: {test_losses_32[-1].item():.2f}')


# ## Batch size = 64

# In[74]:


def load_dataset(train_data, test_data, batch_size):
    torch.manual_seed(42)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


# In[75]:


train_loader2, test_loader2 = load_dataset(train_data, test_data, 64)


# In[76]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.001)
epochs = 8


# In[77]:


train_losses_64, test_losses_64, train_correct_64, test_correct_64 = train_model(CNNmodel, train_loader2, test_loader2, criterion, optimizer, epochs)


# In[78]:


train_accuracies_64 = get_train_accuracies(train_correct_64)
test_accuracies_64 = get_test_accuracies(test_correct_64)


# In[79]:


plot_investigation_losses(train_losses_64, 'CNNmodel with batch size of 64 - training loss', test_losses_64, 'CNNmodel with batch size of 64 - validation loss')


# In[80]:


plot_investigation_accuracies(train_accuracies_64, 'CNNmodel with batch size of 64 - training accuracy', test_accuracies_64, 'CNNmodel with batch size of 64 - validation accuracy')


# In[81]:


print(f'Train accuracy: {train_accuracies_64[-1].item():.2f}%')
print(f'Test accuracy: {test_accuracies_64[-1].item():.2f}%')
print(f'Train loss: {train_losses_64[-1].item():.2f}')
print(f'Test loss: {test_losses_64[-1].item():.2f}')


# ## Batch size = 128

# In[82]:


train_loader3, test_loader3 = load_dataset(train_data, test_data, 128)


# In[83]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.001)
epochs = 8


# In[84]:


train_losses_128, test_losses_128, train_correct_128, test_correct_128 = train_model(CNNmodel, train_loader3, test_loader3, criterion, optimizer, epochs)


# In[85]:


train_accuracies_128 = get_train_accuracies(train_correct_128)
test_accuracies_128 = get_test_accuracies(test_correct_128)


# In[86]:


plot_investigation_losses(train_losses_128, 'CNNmodel with batch size of 128 - training loss', test_losses_128, 'CNNmodel with batch size of 128 - validation loss')


# In[87]:


plot_investigation_accuracies(train_accuracies_128, 'CNNmodel with batch size of 128 - training accuracy', test_accuracies_128, 'CNNmodel with batch size of 128 - validation accuracy')


# In[88]:


print(f'Train accuracy: {train_accuracies_128[-1].item():.2f}%')
print(f'Test accuracy: {test_accuracies_128[-1].item():.2f}%')
print(f'Train loss: {train_losses_128[-1].item():.2f}')
print(f'Test loss: {test_losses_128[-1].item():.2f}')


# # Investigate optimisation techniques

# ## AdaGrad optimizer

# In[132]:


criterion = nn.CrossEntropyLoss()
optimizer_ada = torch.optim.Adagrad(CNNmodel.parameters(), lr=0.001)
epochs = 8


# In[133]:


train_losses_ada, test_losses_ada, train_correct_ada, test_correct_ada = train_model(CNNmodel, train_loader, test_loader, criterion, optimizer_ada, epochs)


# In[134]:


train_accuracies_ada = get_train_accuracies(train_correct_ada)
test_accuracies_ada = get_test_accuracies(test_correct_ada)


# In[135]:


plot_investigation_losses(train_losses_ada, 'CNNmodel with AdaGrad optimizer - training loss', test_losses_ada, 'CNNmodel with AdaGrad optimizer - validation loss')


# In[136]:


plot_investigation_accuracies(train_accuracies_ada, 'CNNmodel with AdaGrad optimizer - training accuracy', test_accuracies_ada, 'CNNmodel with AdaGrad optimizer - validation accuracy')


# In[137]:


print(f'Train accuracy: {train_accuracies_ada[-1].item():.2f}%')
print(f'Test accuracy: {test_accuracies_ada[-1].item():.2f}%')
print(f'Train loss: {train_losses_ada[-1].item():.2f}')
print(f'Test loss: {test_losses_ada[-1].item():.2f}')


# ## RMSProp optimizer

# In[152]:


criterion = nn.CrossEntropyLoss()
optimizer_rmsp = torch.optim.RMSprop(CNNmodel.parameters(), lr=0.001)
epochs = 8


# In[96]:


train_losses_rmsp, test_losses_rmsp, train_correct_rmsp, test_correct_rmsp = train_model(CNNmodel, train_loader, test_loader, criterion, optimizer_rmsp, epochs)


# In[97]:


train_accuracies_rmsp = get_train_accuracies(train_correct_rmsp)
test_accuracies_rmsp = get_test_accuracies(test_correct_rmsp)


# In[98]:


plot_investigation_losses(train_losses_rmsp, 'CNNmodel with RMSProp optimizer - training loss', test_losses_rmsp, 'CNNmodel with RMSProp optimizer - validation loss')


# In[99]:


plot_investigation_accuracies(train_accuracies_rmsp, 'CNNmodel with RMSProp optimizer - training accuracy', test_accuracies_rmsp, 'CNNmodel with RMSProp optimizer - validation accuracy')


# In[100]:


print(f'Train accuracy: {train_accuracies_rmsp[-1].item():.2f}%')
print(f'Test accuracy: {test_accuracies_rmsp[-1].item():.2f}%')
print(f'Train loss: {train_losses_rmsp[-1].item():.2f}')
print(f'Test loss: {test_losses_rmsp[-1].item():.2f}')


# # Investigate the regularization strategy

# ## L2 regularization

# In[101]:


class ConvolutionalNetwork2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1) #3 input color channels, 6 filters (output channels), kernel size 3x3, stride 1, padding tbc
        self.conv2 = nn.Conv2d(6, 16, 3, 1) #6 filters, 16 filters
        self.fc1 = nn.Linear(54*54*16, 120) #120 flattened out neurons
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2) #kernel size 2, stride 2, padding tbc
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54*54*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)


# In[102]:


torch.manual_seed(42)
CNNmodel2 = ConvolutionalNetwork2()
CNNmodel2


# In[103]:


criterion = nn.CrossEntropyLoss()
optimizer_l2 = torch.optim.Adam(CNNmodel2.parameters(), lr=0.001, weight_decay=1e-5)
epochs = 8


# In[104]:


train_losses_l2, test_losses_l2, train_correct_l2, test_correct_l2 = train_model(CNNmodel2, train_loader, test_loader, criterion, optimizer_l2, epochs)


# In[105]:


train_accuracies_l2 = get_train_accuracies(train_correct_l2)
test_accuracies_l2 = get_test_accuracies(test_correct_l2)


# In[106]:


plot_investigation_losses(train_losses_l2, 'CNNmodel with L2 regularization - training loss', test_losses_l2, 'CNNmodel with L2 regularization - validation loss')


# In[107]:


plot_investigation_accuracies(train_accuracies_l2, 'CNNmodel with L2 regularization - training accuracy', test_accuracies_l2, 'CNNmodel with L2 regularization - validation accuracy')


# In[108]:


print(f'Train accuracy: {train_accuracies_l2[-1].item():.2f}%')
print(f'Test accuracy: {test_accuracies_l2[-1].item():.2f}%')
print(f'Train loss: {train_losses_l2[-1].item():.2f}')
print(f'Test loss: {test_losses_l2[-1].item():.2f}')


# ## Dropout rate = 0.5

# In[109]:


class ConvolutionalNetwork3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1) #3 input color channels, 6 filters (output channels), kernel size 3x3, stride 1
        self.conv2 = nn.Conv2d(6, 16, 3, 1) #6 filters, 16 filters
        self.fc1 = nn.Linear(54*54*16, 120) #120 flattened out neurons
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2) #kernel size 2, stride 2
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54*54*16)
        X = self.dropout(X)
        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)


# In[110]:


torch.manual_seed(42)
CNNmodel3 = ConvolutionalNetwork3()
CNNmodel3


# In[111]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNNmodel3.parameters(), lr=0.001)
epochs = 8


# In[112]:


train_losses_05, test_losses_05, train_correct_05, test_correct_05 = train_model(CNNmodel3, train_loader, test_loader, criterion, optimizer, epochs)


# In[113]:


train_accuracies_05 = get_train_accuracies(train_correct_05)
test_accuracies_05 = get_test_accuracies(test_correct_05)


# In[114]:


plot_investigation_losses(train_losses_05, 'CNNmodel with Dropout rate 0.5 - training loss', test_losses_05, 'CNNmodel with Dropout rate 0.5 - validation loss')


# In[115]:


plot_investigation_accuracies(train_accuracies_05, 'CNNmodel with Dropout rate 0.5 - training accuracy', test_accuracies_05, 'CNNmodel with Dropout rate 0.5 - validation accuracy')


# In[116]:


print(f'Train accuracy: {train_accuracies_05[-1].item():.2f}%')
print(f'Test accuracy: {test_accuracies_05[-1].item():.2f}%')
print(f'Train loss: {train_losses_05[-1].item():.2f}')
print(f'Test loss: {test_losses_05[-1].item():.2f}')


# # Use existing models pre-trained

# In[117]:


AlexNetmodel = models.alexnet(pretrained=True)
AlexNetmodel


# In[118]:


for param in AlexNetmodel.parameters():
    param.requires_grad = False


# In[119]:


torch.manual_seed(42)
AlexNetmodel.classifier = nn.Sequential(nn.Linear(9216, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.4),
                                 nn.Linear(1024, 3),
                                 nn.LogSoftmax(dim=1))
AlexNetmodel


# In[120]:


count_parameters(AlexNetmodel)


# In[140]:


criterion_alex = nn.CrossEntropyLoss()
optimizer_alex = torch.optim.Adagrad(AlexNetmodel.classifier.parameters(), lr=0.001)
epochs = 8


# In[141]:


train_losses_alex, test_losses_alex, train_correct_alex, test_correct_alex = train_model(AlexNetmodel, train_loader, test_loader, criterion_alex, optimizer_alex, epochs)


# In[142]:


train_accuracies_alex = get_train_accuracies(train_correct_alex)
test_accuracies_alex = get_test_accuracies(test_correct_alex)


# In[143]:


plot_losses(train_losses_alex, test_losses_alex)


# In[144]:


plot_accuracies(train_accuracies_alex, test_accuracies_alex)


# ## Evaluate model performance

# In[145]:


print(f'Train accuracy: {train_accuracies_alex[-1].item():.2f}%')
print(f'Test accuracy: {test_accuracies_alex[-1].item():.2f}%')
print(f'Train loss: {train_losses_alex[-1].item():.2f}')
print(f'Test loss: {test_losses_alex[-1].item():.2f}')


# ## Display the confusion matrix

# In[146]:


predicted_alex, X_test_alex, y_test_alex = create_cf(test_data, AlexNetmodel)


# ## Save the trained model

# In[147]:


torch.save(AlexNetmodel.state_dict(), 'AlexNetFruitImageCNNModel.pth')


# In[148]:


torch.save(
    {
        'epoch': epochs,
        'model_state_dict': AlexNetmodel.state_dict(),
        'optimizer_state_dict': optimizer_alex.state_dict(),
        'criterion_state_dict': criterion_alex.state_dict(),
    },
    'AlexNetFruitImageCNNModel.pth',
)


# # Loss & Accuracy summary

# In[149]:


input_data = [{"Model": "Initial CNN model",
                "Train loss": round(train_losses[-1].item(),2),
                "Test loss": round(test_losses[-1].item(),2),
                "Train accuracy (%)": round(train_accuracies[-1].item(),2),
                "Test accuracy (%)": round(test_accuracies[-1].item(),2)},
              {"Model": "CNN model with NLL loss function",
              "Train loss": round(train_losses_nll[-1].item(),2),
                "Test loss": round(test_losses_nll[-1].item(),2),
                "Train accuracy (%)": round(train_accuracies_nll[-1].item(),2),
                "Test accuracy (%)": round(test_accuracies_nll[-1].item(),2)},
              {"Model": "CNN model with batch size 32",
               "Train loss": round(train_losses_32[-1].item(),2),
                "Test loss": round(test_losses_32[-1].item(),2),
                "Train accuracy (%)": round(train_accuracies_32[-1].item(),2),
                "Test accuracy (%)": round(test_accuracies_32[-1].item(),2)},
              {"Model": "CNN model with batch size 64",
               "Train loss": round(train_losses_64[-1].item(),2),
                "Test loss": round(test_losses_64[-1].item(),2),
                "Train accuracy (%)": round(train_accuracies_64[-1].item(),2),
                "Test accuracy (%)": round(test_accuracies_64[-1].item(),2)},
              {"Model": "CNN model with batch size 128",
               "Train loss": round(train_losses_128[-1].item(),2),
                "Test loss": round(test_losses_128[-1].item(),2),
                "Train accuracy (%)": round(train_accuracies_128[-1].item(),2),
                "Test accuracy (%)": round(test_accuracies_128[-1].item(),2)},
             {"Model": "CNN model with optimizer AdaGrad",
               "Train loss": round(train_losses_ada[-1].item(),2),
                "Test loss": round(test_losses_ada[-1].item(),2),
                "Train accuracy (%)": round(train_accuracies_ada[-1].item(),2),
                "Test accuracy (%)": round(test_accuracies_ada[-1].item(),2)},
             {"Model": "CNN model with optimizer RMSProp",
               "Train loss": round(train_losses_rmsp[-1].item(),2),
                "Test loss": round(test_losses_rmsp[-1].item(),2),
                "Train accuracy (%)": round(train_accuracies_rmsp[-1].item(),2),
                "Test accuracy (%)": round(test_accuracies_rmsp[-1].item(),2)},
             {"Model": "CNN model with batch L2 Regularization",
               "Train loss": round(train_losses_l2[-1].item(),2),
                "Test loss": round(test_losses_l2[-1].item(),2),
                "Train accuracy (%)": round(train_accuracies_l2[-1].item(),2),
                "Test accuracy (%)": round(test_accuracies_l2[-1].item(),2)},
             {"Model": "CNN model with Dropout rate 0.5",
               "Train loss": round(train_losses_05[-1].item(),2),
                "Test loss": round(test_losses_05[-1].item(),2),
                "Train accuracy (%)": round(train_accuracies_05[-1].item(),2),
                "Test accuracy (%)": round(test_accuracies_05[-1].item(),2)},
             {"Model": "Baseline MLP model",
               "Train loss": round(train_losses_mlp[-1].item(),2),
                "Test loss": round(test_losses_mlp[-1].item(),2),
                "Train accuracy (%)": round(train_accuracies_mlp[-1].item(),2),
                "Test accuracy (%)": round(test_accuracies_mlp[-1].item(),2)},
             {"Model": "Pre-trained AlexNet model",
               "Train loss": round(train_losses_alex[-1].item(),2),
                "Test loss": round(test_losses_alex[-1].item(),2),
                "Train accuracy (%)": round(train_accuracies_alex[-1].item(),2),
                "Test accuracy (%)": round(test_accuracies_alex[-1].item(),2)},]


# In[150]:


summary_df = pd.DataFrame(input_data)
summary_df


# In[ ]:




