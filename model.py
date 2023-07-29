import torch.nn as nn
import torch
import os
import torch.nn.functional as F
import torch.optim as optim



class Linear_QN(nn.Module):

    def __init__(self, input_layer , hidden_layer , output_layer) -> None:
        super(Linear_QN,self).__init__()

        self.fc1 = nn.Linear(input_layer,hidden_layer)
        self.fc2 = nn.Linear(hidden_layer,output_layer)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self,file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainner:
    def __init__(self,lr,gamma,model):
        self.lr = lr
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state,dtype=torch.float)
        next_state = torch.tensor(next_state,dtype=torch.float)
        action = torch.tensor(action,dtype=torch.long)
        reward = torch.tensor(reward,dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )


        prd = self.model(state)

        # Q_new = r+y*max(next_pre)
        target = prd.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new



        self.optimizer.zero_grad()
        loss = self.criterion(target, prd)
        loss.backward()

        self.optimizer.step()
