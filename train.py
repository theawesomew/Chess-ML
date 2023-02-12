import numpy as np
import chess
import torch
import torch.nn as nn
import pickle
from state import Board
from torch.utils.data import DataLoader
from tqdm import trange

class Model (nn.Module):

    def __init__ (self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(257, 180),
            nn.ReLU(),
            nn.Linear(180, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def __call__ (self, x):
        return self.forward(x)

    def forward (self, x):
        return torch.tanh(self.model(x)) 

if __name__ == '__main__':
    m = Model()

    dataset = np.genfromtxt('dataset.csv', delimiter=',')

    standard_board_tensor = torch.from_numpy(Board(chess.Board()).serialize()).float()

    print("Before training valuation of a standard board position: ", m(standard_board_tensor).item())

    for i in range(len(dataset)):
        pos, valuation = torch.from_numpy(dataset[i][:-1]).float(), torch.tensor([dataset[i][-1]], dtype=torch.float32)
        output = m(pos)
        loss = m.loss(output, valuation)
        print("#%i: %.14f" % (i, loss))
        loss.backward()
        m.optim.step()
        m.optim.zero_grad()

    f = open('model.pickle', 'wb')
    f.write(pickle.dumps(m))
    f.close()
    print("After training valuation of a standard board position: ", m(standard_board_tensor).item())
     
