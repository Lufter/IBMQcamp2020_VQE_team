from qiskit.aqua.algorithms import VQE, NumPyEigensolver
import matplotlib.pyplot as plt
import numpy as np
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.circuit.library import EfficientSU2
from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP
from qiskit.aqua.operators import Z2Symmetries
from qiskit import IBMQ, BasicAer, Aer
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import FermionicOperator
from qiskit import IBMQ
from qiskit.aqua import QuantumInstance
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.providers.aer.noise import NoiseModel
import pandas as pd


def get_qubit_op(dist):
    r'''
    qubitOp might be the mysterious Hamitonion.
    '''
    driver = PySCFDriver(atom="Li .0 .0 .0; H .0 .0 " + str(dist),
                         unit=UnitsType.ANGSTROM,
                         charge=0,
                         spin=0,
                         basis='sto3g')
    molecule = driver.run()
    freeze_list = [0]
    remove_list = [-3, -2]
    repulsion_energy = molecule.nuclear_repulsion_energy
    num_particles = molecule.num_alpha + molecule.num_beta
    num_spin_orbitals = molecule.num_orbitals * 2
    remove_list = [x % molecule.num_orbitals for x in remove_list]
    freeze_list = [x % molecule.num_orbitals for x in freeze_list]
    remove_list = [x - len(freeze_list) for x in remove_list]
    remove_list += [
        x + molecule.num_orbitals - len(freeze_list) for x in remove_list
    ]
    freeze_list += [x + molecule.num_orbitals for x in freeze_list]
    ferOp = FermionicOperator(h1=molecule.one_body_integrals,
                              h2=molecule.two_body_integrals)
    ferOp, energy_shift = ferOp.fermion_mode_freezing(freeze_list)
    num_spin_orbitals -= len(freeze_list)
    num_particles -= len(freeze_list)
    ferOp = ferOp.fermion_mode_elimination(remove_list)
    num_spin_orbitals -= len(remove_list)
    qubitOp = ferOp.mapping(map_type='parity', threshold=0.00000001)
    qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)
    shift = energy_shift + repulsion_energy
    return qubitOp, num_particles, num_spin_orbitals, shift


backend = BasicAer.get_backend("statevector_simulator")
distances = np.arange(0.5, 4.0, 0.00035)
exact_energies = []
vqe_energies = []
optimizer = SLSQP(maxiter=5)
labels = []  #([len(distances)])

for dist in distances:
    qubitOp, num_particles, num_spin_orbitals, shift = get_qubit_op(dist)
    
    data = qubitOp.print_details()
    # print(data)
    labels.append(data)


result_x=labels
result_x_signed = []
for r_x in result_x:
    temp = [x[6:-4] for x in r_x.split('\n')][:-1]
    temp_signed = []
    for x in temp:
        if '-' in x:
            temp_signed.append(-1*float(x[1:]))
        else:
            temp_signed.append(float(x[1:]))
    result_x_signed.append(temp_signed)

pd.DataFrame(np.array(result_x_signed)).to_csv("data10000_qubitOp_details.csv")
print("All energies have been calculated")
