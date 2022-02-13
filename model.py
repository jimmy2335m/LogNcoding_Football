"""
Machine Learning:
1. Regression
- Number, quantitative decision
- Potential: numerical value (prediction)
- 한 그래프를 그려 예측하는 방법

2. Classification
- Categorical decision
- Nationality: categorical value (prediction)
- cluster화 시켜 그룹을 예측하는 방법
- Computer vision: 영상 분석, 정해진 Label 안에서 그룹을 예측한다.
"""

"""
Machine Learning
a.k.a. Neural Network

Background
1. 데이터 수 상당히 많음
2. 그에 따라서 그 데이터를 가지고 연산값이 너무 큼 (빅데이터) = computational cost

머신 러닝 도입 이전에는,
Linear regression, quadratic regression, ... 등 y = ax^2 + bx + c // y = ax + b

Equations -> Layer -> Network

Hidden layer:
1. 기계가 알아서 위와 같은 식을 만들어서 최적의 식을 만들어냄
2. 저희는 이 network가 또는 layer가 어떤 식을 사용하고, 어떤 방식으로 output을 만들어내는지 모름.

- Hyperparameter
1. User (저희)가 조정할 수 있는 파라미터 값
2. 조정한 값이 다르면, 결과가 달라지는 경향이 있음
ex) epoch, learning rate, ...

- Paramter
1. model / network에 있는 variable

- Weights
1. neuron 들의 variable

"""

import torch
import torch.nn.functional as F

class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(LinearRegression, self).__init__()
        self.hidden = torch.nn.Linear(inputSize, hiddenSize)
        self.predict = torch.nn.Linear(hiddenSize, outputSize)
    
    def forward(self, x):
        out = self.hidden(x)
        out = F.relu(out) #ReLU: REctified Linear Unit
        out = self.predict(out)
        return out

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_15 = pd.read_csv('./players_15.csv')

#input_columns = ['overall', 'age', 'height_cm', 'weight_kg', 'international_reputation', 'weak_foot', 'skill_moves']
input_columns = ['overall']

x_train = csv_15[input_columns].to_numpy()
y_train = csv_15['potential'].to_numpy()

x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

inputDim = len(input_columns)
hiddenDim = 10
outputDim = 1

"""
Network Structure
Output layer:   o
Hidden layer:o o o o
Input layer:    o
"""
learningRate = 0.00001
epochs = 100
#num of iteration

model = LinearRegression(inputDim, hiddenDim, outputDim)
print(model)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

#MSE: Mean Square Error
#RMSE: Root Mean Square Error

#SGD: Stochastic Gradient Descent

total_loss = []

#training phase
for epoch in range(epochs):
    if torch.cuda.is_available():
        inputs = torch.from_numpy(x_train).float().cuda()
        labels = torch.from_numpy(y_train).float().cuda()
    else:
        inputs = torch.from_numpy(x_train).float()
        labels = torch.from_numpy(y_train).float()

    #optimizer 초기화
    optimizer.zero_grad()

    #현재 모델에 입력값을 대입하여 예측값 생산
    outputs = model(inputs)

    #Loss function을 이용한 loss 계산
    loss = criterion(outputs, labels)

    #Backward Propagation: Loss에 따라 parameter를 얼마만큼 조정해야 되는지 계산
    loss.backward()
    optimizer.step()

    total_loss.append(loss.item())
    print('Epoch {}, Loss {}'.format(epoch, np.round(loss.item(), 5)))

"""""
Converge: loss 값이 더 이사 변하지 않을때
 = 이미 최적화가 되어 loss 값이 최소화되었을 때

Converge 잘못 되었을때:
1. Overfitting:
2. Underfitting:
 """""

with torch.no_grad():
    if torch.cuda.is_available():
        predicted = model(torch.from_numpy(x_train).float().cuda()).cpu().data.numpy()
    else:
        predicted = model(torch.from_numpy(x_train).float()).data.numpy()
    print(predicted)

plt.clf()
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted, '-r', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()

plt.clf()
plt.plot(total_loss, '-b', label='Loss')
plt.legend(loc='best')
plt.show()