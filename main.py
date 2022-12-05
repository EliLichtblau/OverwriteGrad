import torch
from OverwriteGrad.nn import Linear, ReLU, Softmax
import OverwriteGrad.nn.functional as F
from tensorflow.keras.datasets import mnist

(x_train, y_train), (_, _) = mnist.load_data()
x_train, y_train = torch.tensor(x_train).float().reshape(60_000, 784).cuda(), torch.tensor(y_train).long().cuda()
print(f"x_train: {x_train.shape}, y_train: {y_train.shape} ")


model =  torch.nn.Sequential(
    Linear(784, 128),
    #ReLU(),
    #Linear(256, 128),
    #ReLU(),
    #Linear(256, 128),
    #ReLU(),
    #Linear(128, 10),
    #Softmax()
).cuda()

x = Linear(784, 128).to("cuda")

x(x_train)

