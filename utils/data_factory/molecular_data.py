from qiskit.chemistry.drivers import PySCFDriver, UnitsType
import random
from tqdm import tqdm
from pyscf import scf, gto

for i in tqdm(range(10000)):
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'o2.log'
    dist = str(random.uniform(0.5, 4)) 
    mol.atom = 'Li .0 .0 .0; H .0 .0 ' + dist
    mol.basis = 'sto3g'
    mol.build()
    m = scf.RHF(mol)
    print(mol._atom)
    print('E(HF) = %g' % m.kernel())
    f = open('states.txt', "a")
    f.write(dist + ' ' + str(m.kernel())+"\n")



