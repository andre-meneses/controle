import numpy as np
import sympy as sp
from sympy import symbols

def tf_ss(coeffs_num, coeffs_denom, form='control'):
    """
    Transforma uma função de transferência em sua representação em espaço de estados.
    :param coeffs_num: Lista de coeficientes do numerador [b_n, ..., b_1, b_0].
    :param coeffs_denom: Lista de coeficientes do denominador [a_n, ..., a_1, a_0].
    :param form: 'control' para forma canônica controlável e 'observe' para forma canônica observável.
    :return: Matrizes A, B, C, D do sistema em espaço de estados.
    """
    n = len(coeffs_denom) - 1  # Ordem do sistema
    diagonal = np.eye(n-1)
    D = np.array([0])

    if form == 'control':
        alphas = [-1*a for a in coeffs_denom[:0:-1]]
        zeros_column = np.vstack((n-1)*[0])
        A = np.block([[zeros_column, diagonal], alphas])
        B= np.vstack((np.zeros((n-1, 1)), np.array([[1]])))
        C = np.array([[b for b in coeffs_num[::-1]]])
    elif form == 'observe':
        alphas = np.vstack([[-1*a] for a in coeffs_denom[:0:-1]])
        zeros_line = np.zeros((1,n-1))
        A = np.block([np.vstack((zeros_line, diagonal)), alphas])
        B = np.array([[b] for b in coeffs_num[::-1]])
        C = np.hstack((zeros_line, np.array([[1]])))

    return A,B,C,D

def delta_s(polos):
    s = symbols('s')
    poly = 1

    for polo in polos:
        poly = poly*(s - polo)

    return poly.expand()

def input_matrices():
    # Ask for the system order
    n = int(input("Enter the system order: "))

    # Ask for matrix A
    print(f"Enter the entries for matrix A ({n}x{n}) row by row, separated by commas:")
    A_entries = [input(f"Row {i+1}: ").strip() for i in range(n)]
    A = np.array([list(map(float, entry.split(','))) for entry in A_entries])

    # Ask for matrix B
    print(f"Enter the entries for matrix B ({n}x1), one entry per line:")
    B_entries = [float(input(f"Entry {i+1}: ").strip()) for i in range(n)]
    B = np.array(B_entries).reshape(n, 1)

    # Ask for matrix C
    print("Enter the entries for matrix C (1x{n}), separated by commas:")
    C_entries = input("Row: ").strip()
    C = np.array(list(map(float, C_entries.split(',')))).reshape(1, n)

    # Ask for matrix D
    print("Enter the entry for matrix D (1x1):")
    D_entry = float(input("Entry: ").strip())
    D = np.array([[D_entry]])

    # Return the matrices
    return A, B, C, D

def ackermann_observer_gain(A, C, poles):
    n = A.shape[0]
    # Cria a matriz V
    V = C
    for i in range(1, n):
        V = np.vstack((V, C @ np.linalg.matrix_power(A, i)))

    poly_coeffs = np.poly(poles)
    print(poly_coeffs)

    # Cria q_l(A) = A^n + a_{n-1}A^{n-1} + ... + a_1A + a_0I
    q_l_A = np.linalg.matrix_power(A, n)
    for i in range(1, n+1):
        # print(f"poly: {poly_coeffs[i]}, n-i: {n-i}")
        q_l_A += poly_coeffs[i] * np.linalg.matrix_power(A, n-i)

    print("-------")
    sp.pprint(q_l_A)

    # Calcular L = q_l(A)V^{-1}[0 0 ... 1]^T
    V_inv = np.linalg.inv(V)
    print("--------")
    sp.pprint(V_inv)
    e_n = np.zeros((n, 1))
    e_n[-1, 0] = 1
    L = q_l_A @ V_inv @ e_n

    return L

def ackermann_control_gain(A, B, poles):
    n = A.shape[0]
    # Cria a matriz U
    U = B
    for i in range(1, n):
        U = np.hstack((U, np.linalg.matrix_power(A, i) @ B))

    # Polinômio característico desejado com os coeficientes dados pelos polos
    poly_coeffs = np.poly(poles)
    print(poly_coeffs)

    # Cria q_c(A) = A^n + a_{n-1}A^{n-1} + ... + a_1A + a_0I
    q_c_A = np.linalg.matrix_power(A, n)
    # print(poly_coeffs)
    for i in range(1, n+1):
        # print(f"poly: {poly_coeffs[i]}, n-i: {n-i}")
        q_c_A += poly_coeffs[i] * np.linalg.matrix_power(A, n-i)

    print("-------")
    sp.pprint(q_c_A)

    # Calcular K = [0 0 ... 1]U^{-1}q_c(A)
    U_inv = np.linalg.inv(U)
    print("-------")
    sp.pprint(U_inv)
    e_1 = np.zeros((n, 1))
    e_1[-1, 0] = 1
    K = -1*e_1.T @ U_inv @ q_c_A

    return K


