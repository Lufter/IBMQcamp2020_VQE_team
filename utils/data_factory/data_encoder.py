'''
Data encoding

(https://arxiv.org/pdf/quant-ph/0407010.pdf)

'''
from qiskit import *
import numpy as np 
from math import pi
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

def get_angles(x):

    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(
        np.sqrt(x[2] ** 2 + x[3] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    )

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])



def statepreparation(a):
# TODO: Need a qiskit version.

    n_qbit = 2
    qc = QuantumCircuit(n_qbit, n_qbit)
    qc.ry(a[0], 0)

    qc.cx(0,1)
    qc.ry(a[1], 1)
    qc.cx(0,1)
    qc.ry(a[2], 1)
    qc.barrier()
    qc.x(0)
    qc.cx(0,1)
    qc.ry(a[3], 1)
    qc.cx(0,1)
    qc.ry(a[4], 1)
    qc.cx(0,1)
    qc.x(0)
    qc.barrier()
    qc.measure(0,0)
    qc.measure(1,1)
    qc.draw(output = 'mpl')
    plt.show()
    emulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, emulator, shots=1024)
    hist = job.result().get_counts()
    plot_histogram(hist)
    plt.show()

x = [0.53896774, 0.79503606, 0.27826503, 0.0]
statepreparation(get_angles(x))