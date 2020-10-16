import torchvision
from torchvision import datasets, transforms , models
from torch import Tensor, save
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import copy
import tqdm
from PIL import Image
from qiskit import execute
from qiskit.circuit import Parameter, ControlledGate
from qiskit import Aer
import qiskit
import numpy as np
import torch
from torch.autograd import Function
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
#     print("Running on the GPU")
# else:
#     device = torch.device("cpu")
#     print("Running on the CPU")

device = torch.device("cpu")

from qiskit import execute
from qiskit.circuit import Parameter, ControlledGate
from qiskit import Aer
import qiskit
import numpy as np

from tqdm import tqdm

from matplotlib import pyplot as plt
import torchvision.datasets as dset
np.random.seed = 42

NUM_QUBITS = 4
NUM_SHOTS = 1000
SHIFT = np.pi / 2
LEARNING_RATE = 0.01
MOMENTUM = 0.5
SIMULATOR = Aer.get_backend('qasm_simulator')

# create list of all possible outputs of quantum circuit (2**NUM_QUBITS possible)
import itertools


def create_QC_OUTPUTS():
    measurements = list(itertools.product([0, 1], repeat=NUM_QUBITS))
    return [
        ''.join([str(bit) for bit in measurement])
        for measurement in measurements
    ]


QC_OUTPUTS = create_QC_OUTPUTS()
print(QC_OUTPUTS)


class QiskitCircuit():
    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.n_qubits = n_qubits
        self.thetas = {
            k: Parameter('Theta' + str(k))
            for k in range(self.n_qubits)
        }

        all_qubits = [i for i in range(n_qubits)]
        self.circuit.h(all_qubits)
        self.circuit.cx(0, 1)
        self.circuit.cx(2, 3)
        self.circuit.cx(1, 2)
        self.circuit.barrier()
        for k in range(n_qubits):
            self.circuit.ry(self.thetas[k], k)

#         # Apply controlled-unitary
# #         uc=ry(self.theta4, 4).to_gate().control(4)
# #         self.circuit.append(uc, [0,1,2,3,4])
#         self.circuit.ry(self.theta4, 4).to_gate().control(4)

        self.circuit.measure_all()
        # ---------------------------

        self.backend = backend
        self.shots = shots


#             check = perc
#             for i in range(nr_qubits):
#                 check *= (float(key[i])-1/2)*2
#             expects += check

    def N_qubit_expectation_Z(self, counts, shots, nr_qubits):
        expects = np.zeros(len(QC_OUTPUTS))
        for k in range(len(QC_OUTPUTS)):
            key = QC_OUTPUTS[k]
            perc = counts.get(key, 0) / shots
            expects[k] = perc
        return expects

    def run(self, i):
        params = i
        #         print('params = {}'.format(len(params)))
        backend = Aer.get_backend('qasm_simulator')

        job_sim = execute(self.circuit,
                          self.backend,
                          shots=self.shots,
                          parameter_binds=[{
                              self.thetas[k]: params[k].item()
                              for k in range(NUM_QUBITS)
                          }])
        #
        result_sim = job_sim.result()
        counts = result_sim.get_counts(self.circuit)
        return self.N_qubit_expectation_Z(counts, self.shots, NUM_QUBITS)

circuit = QiskitCircuit(NUM_QUBITS, SIMULATOR, NUM_SHOTS)
print('Expected value for rotation [pi/4]: {}'.format(
    circuit.run(torch.Tensor([np.pi / 4] * NUM_QUBITS))))
#circuit.circuit.draw(output='mpl', filename='Figures/{}-qubit circuit ryN.jpg'.format(NUM_QUBITS))


class TorchCircuit(Function):
    @staticmethod
    def forward(ctx, i):
        if not hasattr(ctx, 'QiskitCirc'):
            ctx.QiskitCirc = QiskitCircuit(NUM_QUBITS,
                                           SIMULATOR,
                                           shots=NUM_SHOTS)

        exp_value = ctx.QiskitCirc.run(i)

        result = torch.tensor([exp_value])

        ctx.save_for_backward(result.to(device), i)

        return result

    def backward(ctx, grad_output):

        forward_tensor, i = ctx.saved_tensors
        #         print('forward_tensor = {}'.format(forward_tensor))
        input_numbers = i.to(device)
        #         print('input_numbers = {}'.format(input_numbers))
        gradients = torch.Tensor().to(device)

        for k in range(NUM_QUBITS):
            shift_right = input_numbers.detach().clone()
            shift_right[k] = shift_right[k] + SHIFT
            shift_left = input_numbers.detach().clone()
            shift_left[k] = shift_left[k] - SHIFT

            #             print('shift_right = {}, shift_left = {}'.format(shift_right, shift_left))

            expectation_right = ctx.QiskitCirc.run(shift_right)
            expectation_left = ctx.QiskitCirc.run(shift_left)
            #             print('expectation_right = {}, \nexpectation_left = {}'.format(expectation_right, expectation_left))

            gradient = torch.tensor([expectation_right]) - torch.tensor(
                [expectation_left])
            # rescale gradient
            #             gradient = gradient / torch.norm(gradient)
            #             print('gradient for k={}: {}'.format(k, gradient))
            gradients = torch.cat((gradients, gradient.float().to(device)))


#         print(gradients)

        result = gradients.clone()
        #         print('gradients = {}'.format(result))
        #         print('grad_output = {}'.format(grad_output))

        ret = (result.float() * grad_output.to(device).float()).T.to(device)
        #         print(ret)
        return ret

x = torch.tensor([np.pi / 4] * NUM_QUBITS, requires_grad=True).to(device)

qc = TorchCircuit.apply
y1 = qc(x).to(device)
print('y1 after quantum layer: {}'.format(y1.float()))
y1 = nn.Linear(2**NUM_QUBITS, 1).to(device)(y1.float()).to(device)
y1.backward()
print('x.grad = {}'.format(x.grad))
x = torch.randn(3, 4, 5)

qc = TorchCircuit.apply


def cost(x):
    target = -1
    expval = qc(x)[0]
    # simple linear layer: average all outputs of quantum layer
    #     print(expval)
    val = sum([(i + 1) * expval[i]
               for i in range(2**NUM_QUBITS)]) / 2**NUM_QUBITS
    #     print(val)
    return torch.abs(val - target)**2, expval


x = torch.tensor([-np.pi / 4] * NUM_QUBITS, requires_grad=True)
opt = torch.optim.Adam([x], lr=0.1)

num_epoch = 100

loss_list = []
expval_list = []

for i in tqdm(range(num_epoch)):
    # for i in range(num_epoch):
    opt.zero_grad()
    loss, expval = cost(x)
    loss.backward()
    opt.step()
    loss_list.append(loss.item())
    expval_list.append(expval)

plt.plot(loss_list)
plt.show()

img_dimensions = 300
batch_size = 1

data_dir = './data/'

# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder('D:/Studio_Lufter/Ibmq_camp2020/catdog/data/train', transform=train_transforms)
test_data = datasets.ImageFolder('D:/Studio_Lufter/Ibmq_camp2020/catdog/data/test', transform=test_transforms)

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1)




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.vgg = models.vgg19_bn(pretrained = True)
        for param in self.vgg.features.parameters():
            param.requires_grad = False
        
        self.vgg.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(4096, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(512, 256)
                                 )
        
        #self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        #self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(256, NUM_QUBITS)
        self.qc = TorchCircuit.apply
        self.fc3 = nn.Linear(2**NUM_QUBITS, 2)
        #self.qc = TorchCircuit.apply

    def forward(self, x):
        #x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = x.view(-1, 320)
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.vgg(x)
        x = self.fc2(x)
        x = np.pi * torch.tanh(x)
        #         print('params to QC: {}'.format(x))
        
        x = qc(x[0])  # QUANTUM LAYER
        #         print('output of QC = {}'.format(x))
        x = torch.Tensor(x.float()).to(device)
        x = self.fc3(x)
        #x = qc(x[0])
        #x = torch.Tensor(x.float()).to(device)
        #x.float()
        x = torch.softmax(x, dim=1)
        return x

    #def predict(self, x):
        # apply softmax
        #pred = self.forward(x)
        #         print(pred)
        #ans = torch.argmax(pred[0]).item()
        #return torch.tensor(ans)


network = Net().to(device)
optimizer = optim.Adam(network.parameters(), lr=0.001)

# optimizer = optim.Adam(network.parameters(), lr=learning_rate)

epochs = 20
loss_list = []
loss_func = nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_data_loader):
        data, target = data.to(device), target.to(device)
        #         print(batch_idx)
        optimizer.zero_grad()
        # Forward pass
        #print(data.dtype)
        output = network(data).to(device)
        # Calculating loss
        loss = loss_func(output, target)
        # Backward pass
        loss.backward()
        # Optimize the weights
        optimizer.step()

        total_loss.append(loss.item())

    loss_list.append(sum(total_loss) / len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
        100. * (epoch + 1) / epochs, loss_list[-1]))

plt.plot(loss_list)
plt.title('Hybrid NN Training Convergence for {}-qubit'.format(NUM_QUBITS))
plt.xlabel('Training Iterations')
plt.ylabel('Cross Entropy Loss')
plt.savefig('Figures/{}-qubit Loss Curve ryN.jpg'.format(NUM_QUBITS))

accuracy = 0
number = 0
for batch_idx, (data, target) in enumerate(test_loader):
    number += 1
    output = network.predict(data).item()
    accuracy += (output == target[0].item()) * 1

print("Performance on test data is is: {}/{} = {}%".format(
    accuracy, number, 100 * accuracy / number))

n_samples_shape = (8, 6)
count = 0
fig, axes = plt.subplots(nrows=n_samples_shape[0],
                         ncols=n_samples_shape[1],
                         figsize=(10, 2 * n_samples_shape[0]))

network.eval()
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_data_loader):
        if count == n_samples_shape[0] * n_samples_shape[1]:
            break
        pred = network.predict(data).item()

        axes[count // n_samples_shape[1]][count % n_samples_shape[1]].imshow(
            data[0].numpy().squeeze(), cmap='gray')

        axes[count // n_samples_shape[1]][count %
                                          n_samples_shape[1]].set_xticks([])
        axes[count // n_samples_shape[1]][count %
                                          n_samples_shape[1]].set_yticks([])
        axes[count // n_samples_shape[1]][count %
                                          n_samples_shape[1]].set_title(
                                              'Predicted {}'.format(pred))

        count += 1
