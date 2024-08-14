import numpy as np
import sympy as sp

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
        C = np.array([b for b in coeffs_num[::-1]])
    elif form == 'observe':
        alphas = np.vstack([[-1*a] for a in coeffs_denom[:0:-1]])
        zeros_line = np.zeros((1,n-1))
        A = np.block([np.vstack((zeros_line, diagonal)), alphas])
        B = np.array([[b] for b in coeffs_num[::-1]])
        C = np.hstack((zeros_line, np.array([[1]])))

    return A,B,C,D

