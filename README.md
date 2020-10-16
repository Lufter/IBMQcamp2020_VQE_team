# NTUQ Hackathon
Github issue #10 "[Variational Quantum Eigensolver](https://github.com/qiskit-community/qiskit-hackathon-taiwan-20/issues/10)"

## Topic

## Discussion

* suggestion: establish the topic and make a github repository first

* 可以請Coach把今天提到的論文丟上來嗎

* 如果有人想了解一下Quantum Machine Learining，可以參考[Variational classifier[5]](https://pennylane.ai/qml/demos/tutorial_variational_classifier.html)，它有提到如何把**傳統資料iris dataset encode映射成Qubit的形式**。
"State preparation is not as simple as when we represent a bitstring with a basis state. Every input x has to be translated into a set of angles which can get fed into a small routine for state preparation. To simplify things a bit, we will work with data from the positive subspace, so that we can ignore signs (which would require another cascade of rotations around the z axis)."
文章中也有程式碼。

* 剛剛跟峻瑋討論說，如果用VQE做QNN的話，可以用Pennylane，那可能會需要看一下[PennyLane-Qiskit Plugin](https://github.com/PennyLaneAI/pennylane-qiskit)這個repo跟[doc](https://pennylaneqiskit.readthedocs.io/en/latest/)

* 有人還記得兩個subsystem的Hamiltonian分別訓練的那篇叫甚麼嗎? (Solved: Please view [6])

* 醫療研究中藥物分子的化學表達式(SMILES)
    * 有一個python輕量級**處理SMILES格式**的套件[pysmiles](https://pypi.org/project/pysmiles/)
    * label是甚麼?
    * 問題:如果要用SMILES資料的話，不一定能轉成Hamiltonian

* Variational encoding a way to encode the input classical data into a quantum state.


* optimizer
    * cobyla : gradient-free
    * spsa : noisy-resistant but convergent slower
* [LSTM-meta-learner的解釋](https://wei-tianhao.github.io/blog/2019/09/17/meta-learning.html) 
    
* more entaglement, more powerful
    * explaination: 假如沒有糾纏態(product state)的話，比較像是每個qubit分別預測0~9的二進位表示中的個別位元而不是預測整個數字。
* 二進位表示的label導致數字之間常常只有一個位元有所差異
* 
## Reference


[1] Alberto Peruzzo *etal.*, [*"A variational eigenvalue solver on a quantum processor"*](https://arxiv.org/pdf/1304.3061.pdf),  Nature communications 5, 4213 (2014)

[2] Rongxin Xia and Sabre Kais, ["Hybrid Quantum-Classical Neural Network for Calculating Ground State Energies of Molecules"](https://www.mdpi.com/1099-4300/22/8/828/htm), (2020)

[3] Pennylane(Xanadu),[*"A brief overview of VQE"*](https://pennylane.ai/qml/demos/tutorial_vqe.html).

[4] Pennylane(Xanadu), [*"Accelerating VQEs with quantum natural gradient"*](https://pennylane.ai/qml/demos/tutorial_vqe_qng.html).

[5] Pennylane(Xanadu), [*"Variational classifier"*](https://pennylane.ai/qml/demos/tutorial_variational_classifier.html).

[6] K. Fujii *etal.*, ["*Deep Variational Quantum Eigensolver: a divide-and-conquer method for solving a larger problem with smaller size quantum computers*"](https://arxiv.org/pdf/2007.10917.pdf), (2020)

[7] V. Verteletskyi etal., ["*Measurement Optimization in the Variational Quantum Eigensolver Using a Minimum Clique Cover*"](https://arxiv.org/pdf/1912.06184.pdf), (2020)

[8] Möttönen, *etal.* [*Transformation of quantum states using uniformly controlled rotations*](https://arxiv.org/pdf/quant-ph/0407010.pdf), (2008)]

* https://github.com/BoschSamuel/QizGloria/tree/master/Notebooks
* https://github.com/dumkar/learning-to-learn-qnn
* https://github.com/liangqiyao990210/Quantum-Deep-Learning/tree/master/MNIST01-u3

## Result

### 4-qubit circuit ryN + CNOT

![](https://i.imgur.com/xSsDItd.png)

* training curve

![](https://i.imgur.com/C10onv1.png)

Training [5%]	Loss: 0.6041
Training [10%]	Loss: 0.4719
Training [15%]	Loss: 0.4082
Training [20%]	Loss: 0.3702
Training [25%]	Loss: 0.3507
Training [30%]	Loss: 0.3410
Training [35%]	Loss: 0.3345
Training [40%]	Loss: 0.3281
Training [45%]	Loss: 0.3248
Training [50%]	Loss: 0.3213
Training [55%]	Loss: 0.3199
Training [60%]	Loss: 0.3177
Training [65%]	Loss: 0.3170
Training [70%]	Loss: 0.3172
Training [75%]	Loss: 0.3165
Training [80%]	Loss: 0.3173
Training [85%]	Loss: 0.3165
Training [90%]	Loss: 0.3143
Training [95%]	Loss: 0.3180
Training [100%]	Loss: 0.3146

* accurancy

Performance on test data is is: 398/400 = 99.5%


### 4-qubit circuit ryN



```
├── LiH (Hydrogen-like molecules)
    ├── train
        ├── class1(IIII)
            ├── coefficient
        ├── class2(IIIZ)
            ├── coefficient
        ├── class3 (IIZI)
            ├── coefficient
        ├── ...
    ├── val
    ├── test
 -> save to csv
 
 Implement MLP (doing regression)

```
```flow
st=>start: dictionary of 4-qubit pauli operators (from exact eigensolver)
e=>end: predict energy
op=>operation: MLP (fully connected layer)
op2=>operation: Quantum circuit
op3=>operation: calculate loss
cond=>condition: 是或否？

st->op->op2->op3->e
```
