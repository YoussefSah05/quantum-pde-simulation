# %%
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import UnitaryGate, XGate
from qiskit.circuit.library import Initialize
from qiskit.quantum_info import Statevector
from scipy.linalg import expm
import matplotlib.pyplot as plt
import math

# Parameters
N_x = 2**7
N_t = 100000
velocity = 1
D = 2.2e-5
delta_T = 1 / (N_t - 1)
delta_x = 1 / (N_x - 1)
rh = D * delta_T / (delta_x**2)
ra = velocity * delta_T / delta_x
alpha = (rh + ra / 2) / (1 - 2 * rh)
print(2*alpha)
# Matrices
V = np.array([
    [np.sqrt(1 - 2 * rh), -np.sqrt(2 * rh)],
    [np.sqrt(2 * rh), np.sqrt(1 - 2 * rh)]
], dtype=complex)
V_gate = UnitaryGate(V, label="V")
V_dag = UnitaryGate(V.conj().T, label="V†")

hat_A = np.eye(N_x, dtype=complex)
for i in range(N_x):
    if i > 0:
        hat_A[i, i - 1] = -alpha
    if i < N_x - 1:
        hat_A[i, i + 1] = alpha
hat_A[N_x - 1, 0] = -alpha
hat_A[0, N_x - 1] = alpha

H = np.block([
    [np.zeros((N_x, N_x), dtype=complex), -1j * hat_A.conj().T],
    [1j * hat_A, np.zeros((N_x, N_x), dtype=complex)]
])
U = expm(-1j * np.pi / 2 * H)
U_gate = UnitaryGate(U, label="exp(-iπ/2·H)")

S = np.roll(np.eye(N_x), 1, axis=1)
S_gate = UnitaryGate(S, label="S")

# Registers
n_data = math.ceil(np.log2(N_x))
anc = QuantumRegister(2, 'anc')
data = QuantumRegister(n_data, 'phi_t')
cl = ClassicalRegister(2, 'c')

# Build evolution circuit once (excluding initialization)
base_circuit = QuantumCircuit(anc, data, cl)
base_circuit.append(V_gate, [anc[0]])
base_circuit.append(U_gate.control(1, ctrl_state='0'), [anc[0], anc[1]] + data[:])
base_circuit.append(XGate().control(1, ctrl_state='0'), [anc[0], anc[1]])
base_circuit.append(S_gate.control(), [anc[0]] + data[:])
base_circuit.append(V_dag, [anc[0]])
base_circuit.measure(anc[0], cl[0])
base_circuit.measure(anc[1], cl[1])

# Initial state
x_values = np.linspace(0, 1, N_x)
phi_0 = np.sin(2*np.pi * x_values)
phi_0 = phi_0 / np.linalg.norm(phi_0)
phi_t_list = [phi_0]

backend = Aer.get_backend('statevector_simulator')

# Time evolution
for t in range(1, 3):  # Change to N_t for full time evolution
    init_gate = Initialize(phi_0).gates_to_uncompute().inverse()  # safe state init
    circ = QuantumCircuit(anc, data, cl)
    circ.append(init_gate, data)
    full_circuit = circ.compose(base_circuit, front=False)
    compiled = transpile(full_circuit, backend)
    job = backend.run(compiled)
    result = job.result()
    statevector = result.get_statevector()
    # Post-selection: ancilla == |00⟩
    phi_t = []
    for i, amp in enumerate(statevector):
        bin_str = format(i, f'0{2 + n_data}b')  # total bits = anc + data
        if bin_str[:2] == '00':
            phi_t.append(amp)

    phi_t = np.array(phi_t)[:N_x]
    phi_t = phi_t / np.linalg.norm(phi_t)
    phi_t_list.append(phi_t.copy())
    phi_0 = phi_t  # update for next iteration
print(phi_t)

# Plot analytic and numerical solution
plt.figure(figsize=(10, 6))
for t in range(3):
    time = t*delta_T
    plt.plot(x_values, phi_t_list[t], label=f't = {t}')
    phi_analytic = np.exp(-4 * np.pi**2 * D * time) * np.sin(2 * np.pi * (x_values - velocity * time))
    phi_analytic = phi_analytic / np.linalg.norm(phi_analytic)
    plt.plot(x_values, phi_analytic, '--', label=f'Analytic $t = {t}$')
plt.xlabel("Position $x$")
plt.ylabel("$\phi(x)$")
plt.title("Évolution de $\phi_t(x)$ dans l'espace (analytique vs numérique)")
plt.grid(True)
plt.legend()
plt.show()
full_circuit.draw('mpl')
#%%