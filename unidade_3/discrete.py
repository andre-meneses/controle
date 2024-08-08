import control as ctrl
import numpy as np
import sympy as sp

def discretize(A, B, C, D, Ts, method='zoh'):
    """
    Discretizes a continuous-time state-space model.

    Parameters:
    - A, B, C, D: Arrays representing the state-space model matrices.
    - Ts: Sampling time.
    - method: Method of discretization ('zoh', 'foh', etc.).

    Returns:
    - Discrete-time state-space model.
    """
    sys_continuous = ctrl.StateSpace(A, B, C, D)
    sys_discrete = ctrl.sample_system(sys_continuous, Ts, method=method)
    return sys_discrete

def check_stability(sys_discrete):
    """
    Checks the stability of a discrete-time state-space model and prints the eigenvalues.

    Parameters:
    - sys_discrete: Discrete-time state-space model.

    Returns:
    - Boolean indicating if the system is stable.
    """
    A_discrete = sys_discrete.A
    eigenvalues = np.linalg.eigvals(A_discrete)
    print("Eigenvalues of the system matrix A:", eigenvalues)

    # A discrete-time system is stable if all eigenvalues have modulus less than 1
    is_stable = np.all(np.abs(eigenvalues) < 1)
    return is_stable

def check_controllability(A, B):
    """
    Computes the controllability matrix of a system and checks if the system is controllable.

    Parameters:
    - A, B: Arrays representing the state matrix and input matrix of the system.

    Returns:
    - Boolean indicating if the system is controllable.
    """
    cont_matrix = ctrl.ctrb(A, B)
    rank_cont_matrix = np.linalg.matrix_rank(cont_matrix)
    print("Controllability Matrix:\n", cont_matrix)
    print("Rank of the Controllability Matrix:", rank_cont_matrix)

    # The system is controllable if the rank of the controllability matrix equals the number of states (rows of A)
    is_controllable = rank_cont_matrix == A.shape[0]
    return is_controllable

def check_observability(A, C):
    """
    Computes the observability matrix of a system and checks if the system is observable.

    Parameters:
    - A, C: Arrays representing the state matrix and output matrix of the system.

    Returns:
    - Boolean indicating if the system is observable.
    """
    obs_matrix = ctrl.obsv(A, C)
    rank_obs_matrix = np.linalg.matrix_rank(obs_matrix)
    print("Observability Matrix:\n", obs_matrix)
    print("Rank of the Observability Matrix:", rank_obs_matrix)

    # The system is observable if the rank of the observability matrix equals the number of states (rows of A)
    is_observable = rank_obs_matrix == A.shape[0]
    return is_observable

def ackermann_state_feedback(A, B, desired_poles):
    """
    Computes the state feedback gain matrix K using Ackermann's formula to place
    the poles of the closed-loop system at specified locations.

    Parameters:
    - A, B: Arrays representing the state matrix and input matrix of the system.
    - desired_poles: List or array of desired eigenvalues for the closed-loop system.

    Returns:
    - K: State feedback gain matrix.
    """
    # Check system controllability
    cont_matrix = ctrl.ctrb(A, B)
    if np.linalg.matrix_rank(cont_matrix) != A.shape[0]:
        raise ValueError("The system is not controllable and pole placement cannot be performed.")

    # Use control.acker to calculate the state feedback gain matrix K
    K = ctrl.acker(A, B, desired_poles)
    return K

def compute_observer_gain(A, C, desired_observer_poles):
    """
    Computes the observer gain matrix L for state estimation.

    Parameters:
    - A, B: Arrays representing the state matrix and output matrix of the system.
    - desired_observer_poles: List or array of desired eigenvalues for the observer system.

    Returns:
    - L: Observer gain matrix.
    """
    # Check system observability
    obs_matrix = ctrl.obsv(A, C)
    if np.linalg.matrix_rank(obs_matrix) != A.shape[0]:
        raise ValueError("The system is not observable and observer design cannot be performed.")

    # Calculate the observer gain matrix L using pole placement
    # Note: ctrl.place_poles is used for pole placement. It works for both state feedback and observer design.
    # We need to transpose A and C because the dual of the state-space is used for observer design.
    L = ctrl.place_poles(A.T, C.T, desired_observer_poles).gain_matrix.T

    return L

def compute_k2_k1(K_hat, A, B, C):
    """
    Computes the feedback components k2 and k1 using a predefined matrix formula,
    where G and H are replaced by the discrete-time system matrices A and B.

    Parameters:
    - K_hat: Primary feedback gain matrix.
    - A, B: Discrete-time state matrix and input matrix of the system.
    - C: Output matrix.

    Returns:
    - k2, k1: Components of the feedback system.
    """
    n = A.shape[0]  # Number of states
    m = B.shape[1]  # Number of inputs

    # Constructing the selector matrix based on the dimensions of A and B
    selector = np.zeros((n + m, m))
    selector[-m:, :] = np.eye(m)  # Set the last block to identity of size m

    # Constructing the inverse of the block matrix
    block_matrix = np.block([
        [A - np.eye(n), B],
        [C @ A, C @ B]
    ])
    inv_block_matrix = np.linalg.inv(block_matrix)

    # Multiplying K_hat with the selector and the inverse of the block matrix
    result = K_hat @ selector @ inv_block_matrix

    # Extract k2 and k1 (assuming result is structured correctly)
    k2 = result[:, :n]  # Feedback gains for state variables
    k1 = result[:, n:]  # Feedback gains for input variables

    return k2, k1

if __name__ == '__main__':
    A = np.array([[0, 1],[0,-2]])
    B = np.array([[0],[1]])
    C = np.array([[1, 0], [0,0]])
    D = np.array([[0],[0]])
    Ts = 1

    sys_discrete = discretize(A, B, C, D, Ts)

    stability = check_stability(sys_discrete)
    print("Is the system stable?", stability)

    controllable = check_controllability(A, B)
    print("Is the system controllable?", controllable)

    # Check observability
    observable = check_observability(A, C)
    print("Is the system observable?", observable)
    

