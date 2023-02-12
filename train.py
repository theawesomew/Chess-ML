import numpy as np
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
from state import Board
from tqdm import trange

class Model (nn.Module):

    def __init__ (self):
        super(Model, self).__init__()

        self.a1 = nn.Conv2d(5, 16, kernel_size=3, padding=1)
        self.a2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.a3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)

        self.b1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        self.c1 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c2 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c3 = nn.Conv2d(64, 128, kernel_size=2, stride=2)

        self.d1 = nn.Conv2d(128, 128, kernel_size=1)
        self.d2 = nn.Conv2d(128, 128, kernel_size=1)
        self.d3 = nn.Conv2d(128, 128, kernel_size=1)

        self.last = nn.Linear(128, 1)

    def __call__ (self, x):
        return self.forward(x)

    def forward (self, x):
        x = F.relu(self.a1(x))
        x = F.relu(self.a2(x))
        x = F.relu(self.a3(x))

        # 4x4
        x = F.relu(self.b1(x))
        x = F.relu(self.b2(x))
        x = F.relu(self.b3(x))

        # 2x2
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))

        # 1x128
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))

        x = x.view(-1, 128)
        x = self.last(x)

        # value output
        return F.tanh(x)         

if __name__ == '__main__':
    model = Model()
    floss = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters())

    dataset = np.genfromtxt('dataset.csv', delimiter=',')

    standard_board_tensor = torch.from_numpy(Board(chess.Board()).serialize()).float()
    
    losses = []

    print("Before training valuation of a standard board position: ", model(standard_board_tensor).item())

    for i in range(len(dataset)):
        optim.zero_grad()
        pos, valuation = torch.from_numpy(dataset[i][:-1]).float(), torch.tensor([dataset[i][-1]], dtype=torch.float32)
        output = model(pos)
        loss = floss(output, valuation)
        print("#%i: %.14f" % (i, loss))
        losses.append(loss.item())
        loss.backward()
        optim.step()

    f = open('model.pickle', 'wb')
    f.write(pickle.dumps(model))
    f.close()
    print("After training valuation of a standard board position: ", model(standard_board_tensor).item())

    x = np.arange(0, len(losses))
    y = np.array(losses)

    fig, ax = plt.subplots(figsize=(max(losses),len(losses)))
    ax.plot(x, y)
    ax.set_title('Loss')
    fig.set_facecolor('lightsteelblue')     
    plt.show()