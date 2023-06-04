import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random
import numpy as np
from collections import deque
from snake_gameai import SnakeGameAI, Direction, Point, BLOCK_SIZE
from Helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# Check MPS availability and enable it
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
else:
    mps_device = torch.device("mps")  # Set the MPS device

# 1. Creating a class named Linear_Qnet for initializing the linear neural network.
# 2. The function forward is used to take the input(11 state vector) and pass it through the 
#    Neural network and apply relu activation function and give the output back i.e the next 
#    move of 1 x 3 vector size. In short, this is the prediction function that would be called by the agent.
# 3. The save function is used to save the trained model for future use.

class Linear_QNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size).to(mps_device)
        self.linear2 = nn.Linear(hidden_size,output_size).to(mps_device)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = '/Users/cvasev/Developer/QLearningSnake'
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(), file_name)
    
    def load(self, file_name='850-epoch.pth'):
        model_folder_path = '/Users/cvasev/Developer/QLearningSnake'
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))

# 1. Initialising QTrainer class
#    ∗ setting the learning rate for the optimizer.
#    * Gamma value that is the discount rate used in Bellman equation.
#    * initialising the Adam optimizer for updation of weight and biases.
#    * criterion is the Mean squared loss function.
# 2. Train_step function 
#    * As you know that PyTorch work only on tensors, so we are converting all the input
#     to tensors.
#    * As discussed above we had a short memory training then we would only pass one value
#     of state, action, reward, move so we need to convert them into a vector, so we had used
#     unsqueezed function .
#    * Get the state from the model and calculate the new Q value using the below formula:
#                    Q_new = reward + gamma * max(next_predicted Qvalue)
#    * calculate the mean squared error between the new Q value and previous Q value and 
#    backpropagate that loss for weight updation. 

class QTrainer:
    def __init__(self,model,lr,gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model.to(mps_device)
        self.optimer = optim.Adam(model.parameters(),lr = self.lr)    
        self.criterion = nn.MSELoss()

    def train_step(self,state,action,reward,next_state,done):
        state = torch.from_numpy(np.array(state, dtype=np.float32)).to(mps_device)
        next_state = torch.from_numpy(np.array(next_state, dtype=np.float32)).to(mps_device)
        action = torch.from_numpy(np.array(action, dtype=np.int64)).to(mps_device)  # Use np.int64 here
        reward = torch.from_numpy(np.array(reward, dtype=np.float32)).to(mps_device)

        if(len(state.shape) == 1): 
            state = torch.unsqueeze(state,0).to(mps_device)
            next_state = torch.unsqueeze(next_state,0).to(mps_device)
            action = torch.unsqueeze(action,0).to(mps_device)
            reward = torch.unsqueeze(reward,0).to(mps_device)
            done = (done, )

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new 

        self.optimer.zero_grad()
        loss = self.criterion(target,pred)
        loss.backward()
        self.optimer.step()
