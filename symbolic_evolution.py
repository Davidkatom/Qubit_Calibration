import random

import numpy as np
from sympy import Matrix, symbols, exp, I, simplify, sqrt, expand_complex, lambdify
from sympy.physics.quantum import Dagger
from sympy.physics.quantum import TensorProduct
import itertools


def zero_state(n):
    if n == 1:
        return Matrix([1, 0])
    return TensorProduct(zero_state(n - 1), zero_state(1))


def one_state(n):
    if n == 1:
        return Matrix([0, 1])
    return TensorProduct(one_state(n - 1), one_state(1))


def plus_state(n):
    if n == 1:
        return (1 / sqrt(2)) * Matrix([1, 1]).T
    return (1 / sqrt(2)) * TensorProduct(plus_state(n - 1), plus_state(1))


def Ry(thetas):
    n = len(thetas)
    if n == 1:
        return Matrix([[np.cos(thetas[0] / 2), -np.sin(thetas[0] / 2)], [np.sin(thetas[0] / 2), np.cos(thetas[0] / 2)]])
    return TensorProduct(Ry(thetas[:-1]), Ry(thetas[-1:]))


def Rz(thetas):
    n = len(thetas)
    if n == 1:
        return Matrix([[exp(-I * thetas[0] / 2), 0], [0, exp(I * thetas[0] / 2)]])
    return TensorProduct(Rz(thetas[:-1]), Rz(thetas[-1:]))


def Rx(thetas):
    n = len(thetas)
    if n == 1:
        return Matrix(
            [[np.cos(thetas[0] / 2), -I * np.sin(thetas[0] / 2)], [-I * np.sin(thetas[0] / 2), np.cos(thetas[0] / 2)]])
    return TensorProduct(Rx(thetas[:-1]), Rx(thetas[-1:]))


def random_state(state):
    n = np.log2(len(state)).astype(int)
    thetasY = [random.uniform(0.25 * np.pi, 1.75 * np.pi) for _ in range(n)]
    # thetasX = [random.uniform(0, 2 * np.pi) for _ in range(n)]
    thetasZ = [random.uniform(0, 2 * np.pi) for _ in range(n)]
    state = apply_operator(state, Ry(thetasY))
    state = apply_operator(state, Rz(thetasZ))
    return state


def evolve_state(state):
    n = np.log2(len(state)).astype(int)
    t = symbols('t', real=True)
    ω = symbols(f'ω0:{n}', real=True)
    j = symbols(f'j0:{n - 1}', real=True)

    basis_states = [''.join(seq) for seq in itertools.product('01', repeat=n)]

    def get_factor(base_state):
        evolution_factor = 1
        for i, qubit in enumerate(base_state):
            if qubit == '1':
                evolution_factor *= exp(I * ω[i] * t)
                if i < len(base_state) - 1 and base_state[i + 1] == '1':
                    evolution_factor *= exp(I * j[i] * t)
        return evolution_factor

    evolved_state_factors = [state[i] * get_factor(basis_states[i]) for i in range(len(state))]
    psi_t = Matrix([evolved_state_factors]).T  # Transpose to make it a column vector
    return psi_t


def apply_operator(state, operator):
    return simplify(expand_complex(operator * state))


def H(n):
    if n == 1:
        return Matrix([[1, 1], [1, -1]]) / sqrt(2)
    return TensorProduct(H(n - 1), H(1))


def Pauli(string):
    n = len(string)
    if n == 1:
        if string == 'X':
            return Matrix([[0, 1], [1, 0]])
        if string == 'Y':
            return Matrix([[0, I], [-I, 0]])
        if string == 'Z':
            return Matrix([[1, 0], [0, -1]])
        if string == 'I':
            return Matrix([[1, 0], [0, 1]])
    return TensorProduct(Pauli(string[0]), Pauli(string[1:]))


def apply_decay(value, decay_rate, n):
    a = symbols(f'a0:{n}', real=True)


def GHZ(n):
    return (1 / sqrt(2)) * (zero_state(n) + one_state(n))


def expectation_value(state, pauli_string, decay=False):
    n = np.log2(len(state)).astype(int)
    a = symbols(f'a0:{n}', real=True)
    t = symbols('t', real=True)
    operator = Pauli(pauli_string)
    decay_factor = 1
    for i in range(len(pauli_string)):
        if pauli_string[i] == 'I':
            continue
        if decay:
            decay_factor *= exp(-a[i] * t)
    return simplify(expand_complex(decay_factor * (Dagger(state) * operator * state)))[0]


def set_parameters(expr, W=None, J=None, A=None, t=None):
    symb = list(expr.free_symbols)
    symbols_w = [symbol for symbol in symb if 'ω' in str(symbol)]
    symbols_j = [symbol for symbol in symb if 'j' in str(symbol)]
    symbols_a = [symbol for symbol in symb if 'a' in str(symbol)]
    symbolT = symbols('t', real=True)

    if t is not None:
        expr = expr.subs(symbolT, t)

    if W is not None:
        W = W[::-1]
        for symbol in symbols_w:
            expr = expr.subs(symbol, W[int(str(symbol)[1])])

    if J is not None:
        J = J[::-1]
        for symbol in symbols_j:
            expr = expr.subs(symbol, J[int(str(symbol)[1])])
    if A is not None:
        A = A[::-1]
        for symbol in symbols_a:
            expr = expr.subs(symbol, A[int(str(symbol)[1])])

    if expr.is_Number:
        # Convert to a Python numerical type if possible
        return float(expr)
    else:
        # Return the symbolic expression if still symbolic
        return expr


def to_func(expr):
    symbols = list(expr.free_symbols)
    return lambdify(symbols, expr, 'numpy')


def get_n_nearest_neighbors(qubits, basis, neighbors=0):
    # This function generates Pauli strings for single qubits and n nearest neighbors
    pauli_strings = []

    # Single qubit measurements
    for i in range(qubits):
        pauli_str = 'I' * i + basis + 'I' * (qubits - i - 1)
        pauli_strings.append(pauli_str)

    # n nearest neighbor pairs
    for distance in range(1, neighbors + 1):
        for i in range(qubits - distance):
            pauli_str = 'I' * i + basis + 'I' * (distance - 1) + basis + 'I' * (qubits - i - distance - 1)
            pauli_strings.append(pauli_str)

    return pauli_strings


def get_nth_nearest_neighbors(qubits, basis, neighbors=0):
    # This function generates Pauli strings for single qubits and n nearest neighbors
    pauli_strings = []  # n nearest neighbor pairs
    for i in range(qubits - neighbors):
        pauli_str = 'I' * i + basis + 'I' * (neighbors - 1) + basis + 'I' * (qubits - i - neighbors - 1)
        pauli_strings.append(pauli_str)

    return pauli_strings


def get_expectation_values_exp(n, neighbors=0, rot=None):
    operators = get_n_nearest_neighbors(n, 'X', neighbors) + get_n_nearest_neighbors(n, 'Y', neighbors)

    state = zero_state(n)
    if rot is None:
        state = apply_operator(state, H(n))
    else:
        state = apply_operator(state, Ry([rot] * n))
    state = evolve_state(state)
    expectation_values = [expectation_value(state, operator) for operator in operators]
    return expectation_values


def get_expectation_of_nth(n, neighbors=0, rot=None):

    operators = get_nth_nearest_neighbors(n, 'X', neighbors) + get_nth_nearest_neighbors(n, 'Y', neighbors)

    state = zero_state(n)
    if rot is None:
        state = apply_operator(state, H(n))
    else:
        state = apply_operator(state, Ry([rot] * n))
    state = evolve_state(state)
    expectation_values = [expectation_value(state, operator) for operator in operators]
    return expectation_values


def get_expectation_values(n, W, J, A, neighbors=0):
    expectation_values = get_expectation_values_exp(n, neighbors)
    expectation_values = [set_parameters(expr, W, J, A) for expr in expectation_values]
    return [to_func(expr) for expr in expectation_values]


def sort_key(parameter):
    prefix_order = {'a': 1, 'ω': 2, 'j': 3}  # Define the order for each prefix
    # Return a tuple that consists of the predefined order and the parameter to ensure stable sorting
    return prefix_order.get(parameter[0].lower(), 4), parameter


def minimize_functions(n, times, neighbors=0):
    symbolT = symbols('t', real=True)

    all_functions = []
    functions = get_expectation_values_exp(n, neighbors)
    for time in times:
        for func in functions:
            func = func.subs(symbolT, time)
            all_functions.append(func)

    # parameters = [all_functions[i].free_symbols for i in range(len(all_functions))]
    # parameters = set.union(*parameters)
    # parameters = [str(parameter) for parameter in parameters]
    # parameters = sorted(parameters, key=sort_key)

    return all_functions

#
# W = [1, 1, 1]
# J = [1, 1]
# A = [1, 1, 1]
#
# functions = minimize_functions(3, [0, 1, 2], 0)
# functions_t = [set_parameters(expr, W, J, A) for expr in functions]
# print(functions_t)
