import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns

# Import data
iris = load_iris()
X = iris.data
y = iris.target

#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

#Convert X_train and X_test to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#Dataset
class IrisData(Dataset):
    def __init__(self, X_train, y_train):
        super().__init__()
        self.X = torch.from_numpy(X_train)
        self.y = torch.from_numpy(y_train)
        self.y = self.y.type(torch.LongTensor)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.len
    
#DataLoader
iris_data = IrisData(X_train=X_train, y_train=y_train)
train_loader = DataLoader(dataset=iris_data, batch_size=32)

#Check for dimension
print('Shape: {}, y shape: {}'.format(iris_data.X.shape, iris_data.y.shape))


#Define Custom MultiClass
class MultiClassNet(nn.Module):
    def __init__(self,  NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES):
        super().__init__()
        self.linear1 = nn.Linear(NUM_FEATURES, HIDDEN_FEATURES)
        self.linear2 = nn.Linear(HIDDEN_FEATURES, NUM_CLASSES)
        self.log_softmax = nn.LogSoftmax(dim=1)

    #Create forward function
    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = self.log_softmax(x)
        return x
    
#Hyperparameter
NUM_FEATURES = iris_data.X.shape[1]
HIDDEN = 6
NUM_CLASSES = len(iris_data.y.unique())
#print(NUM_CLASSES)

#Create model instance
model = MultiClassNet(NUM_FEATURES=NUM_FEATURES, NUM_CLASSES=NUM_CLASSES, HIDDEN_FEATURES=HIDDEN)

#Loss function
criterion = nn.CrossEntropyLoss()

#Optimizer
lr = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

#Training Loop
NUM_EPOCHS = 100
losses = []

for epoch in range(NUM_EPOCHS):
    for X, y in train_loader:

        # initialize gradients
        optimizer.zero_grad()

        #Forward pass
        y_pred_log = model(X)

        #Calculate losses
        loss = criterion(y_pred_log, y)

        #Calculate gradient
        loss.backward()

        #Update parameters
        optimizer.step()

    losses.append(float(loss.data.detach().numpy()))
    #print(losses)

#Plot losses against epochs
print(sns.lineplot(x= range(len(losses)), y = losses))