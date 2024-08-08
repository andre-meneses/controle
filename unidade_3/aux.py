import control as ctrl

import numpy as np

def tf_ss(num, den, form='controllable'):
    """
    Converts a transfer function defined by numerator (num) and denominator (den)
    coefficients to its state-space representation in either the canonical controllable 
    or observable form.

    Parameters:
    num (list): Numerator coefficients.
    den (list): Denominator coefficients.
    form (str): 'controllable' or 'observable' specifying the form of the state-space.

    Returns:
    tuple: Returns the matrices A, B, C of the state-space representation.
    """
    # Normalize the denominator coefficients such that the highest degree coefficient is 1
    den = np.array(den) / den[0]
    num = np.array(num) / den[0]
    
    # Degree of the polynomial
    n = len(den) - 1
    m = len(num) - 1
    
    # Define matrices A, B, C based on the form
    if form.lower() == 'controllable':
        # Controllable form
        A = np.zeros((n, n))
        A[-1, :] = -den[1:]  # Last row is the negative of the denominator coefficients
        A[:-1, 1:] = np.eye(n-1)  # Upper part is the shifted identity matrix
        
        B = np.zeros((n, 1))
        B[-1, 0] = 1
        
        C = np.zeros((1, n))
        C[0, :m+1] = num[::-1]  # Place the numerator coefficients in reverse order

    elif form.lower() == 'observable':
        # Observable form
        A = np.zeros((n, n))
        A[:, 0] = -den[1:]  # First column is the negative of the denominator coefficients
        A[1:, :-1] = np.eye(n-1)  # Lower part is the shifted identity matrix

        C = np.zeros((1, n))
        C[0, 0] = 1
        
        B = np.zeros((n, 1))
        B[:m+1, 0] = num[::-1]  # Place the numerator coefficients in reverse order

    else:
        raise ValueError("Invalid form. Use 'controllable' or 'observable'.")

    return A, B, C

