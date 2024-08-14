import control as ctrl
import numpy as np
import sympy as sp

class DiscreteSystemAnalysis:
    def __init__(self, A, B, C, D, Ts=None):
        self.Ts = Ts

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        # Discretize only if Ts is provided
        self.sys_discrete = self.discretize() if Ts is not None else ctrl.StateSpace(A, B, C, D)

        
    def discretize(self, method='zoh'):
        """
        Discretizes the continuous-time state-space model using Zero-Order Hold by default.
        """
        sys_continuous = ctrl.StateSpace(self.A, self.B, self.C, self.D)
        sys_discrete = ctrl.sample_system(sys_continuous, self.Ts, method=method)

        self.A = sys_discrete.A
        self.B = sys_discrete.B
        self.C = sys_discrete.C
        self.D = sys_discrete.D

        return sys_discrete

    def check_stability(self):
        """
        Checks the stability of the discrete-time state-space model.
        """
        A_discrete = self.sys_discrete.A
        eigenvalues = np.linalg.eigvals(A_discrete)
        print("Eigenvalues of the system matrix A:")
        sp.pprint(sp.Matrix(eigenvalues))

        # A discrete-time system is stable if all eigenvalues have modulus less than 1
        is_stable = np.all(np.abs(eigenvalues) < 1)
        return is_stable

    def check_controllability(self):
        """
        Computes the controllability matrix of the system.
        """
        cont_matrix = ctrl.ctrb(self.A, self.B)
        rank_cont_matrix = np.linalg.matrix_rank(cont_matrix)
        print("Controllability Matrix:")
        sp.pprint(sp.Matrix(cont_matrix))
        print(f"Rank of the Controllability Matrix: {rank_cont_matrix}")

        # The system is controllable if the rank of the controllability matrix equals the number of states
        return rank_cont_matrix == self.A.shape[0]

    def check_observability(self):
        """
        Computes the observability matrix of the system.
        """
        obs_matrix = ctrl.obsv(self.A, self.C)
        rank_obs_matrix = np.linalg.matrix_rank(obs_matrix)
        print("Observability Matrix:")
        sp.pprint(sp.Matrix(obs_matrix))
        print(f"Rank of the Observability Matrix: {rank_obs_matrix}")

        # The system is observable if the rank of the observability matrix equals the number of states
        return rank_obs_matrix == self.A.shape[0]

    def ackermann_state_feedback(self, desired_poles, augmented=False):
        """
        Computes the state feedback gain matrix K using Ackermann's formula.
        """
        cont_matrix = ctrl.ctrb(self.A, self.B)
        if np.linalg.matrix_rank(cont_matrix) != self.A.shape[0]:
            raise ValueError("The system is not controllable and pole placement cannot be performed.")

        K = ctrl.acker(self.A, self.B, desired_poles)
        return K

    def compute_observer_gain(self, desired_observer_poles):
        """
        Computes the observer gain matrix L.
        """
        obs_matrix = ctrl.obsv(self.A, self.C)
        if np.linalg.matrix_rank(obs_matrix) != self.A.shape[0]:
            raise ValueError("The system is not observable and observer design cannot be performed.")

        L = ctrl.place(self.A.T, self.C.T, desired_observer_poles)
        return L.T


    def augment_system(self):
        """
        Augments a state-space system for reference tracking (step or ramp).

        Parameters:
        - reference_type: Type of reference to track ('step' or 'ramp').

        Returns:
        - A_aug: Augmented state matrix.
        - B_aug: Augmented input matrix.
        - C_aug: Augmented output matrix.
        """
        G = self.A
        H = self.B
        C = self.C
        # Original system dimensions

        n = G.shape[0]  # Number of states
        m = H.shape[1]  # Number of inputs
        p = C.shape[0]  # Number of outputs

        # Augmenting state matrix A
        A_aug = np.block([
            [G, H],
            [np.zeros((1,n)), np.zeros((p, m))]
        ])

        B_aug = np.vstack((np.zeros((n, 1)), np.array([[1]])))

        self.A_aug = A_aug
        self.B_aug = B_aug


if __name__ == '__main__':
    A = np.array([[0, 1], [0, -2]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0], [0, 0]])
    D = np.array([[0], [0]])
    Ts = 1  # Sampling time

    # Example for discrete system
    system = SystemAnalysis(A, B, C, D, Ts)
    print("Is the system stable?", system.check_stability())
    print("Is the system controllable?", system.check_controllability())
    print("Is the system observable?", system.check_observability())

    # For already discrete systems, set Ts=None
    discrete_system = SystemAnalysis(A, B, C, D)
    print("Is the discrete system stable?", discrete_system.check_stability())
    print("Is the discrete system controllable?", discrete_system.check_controllability())
    print("Is the discrete system observable?", discrete_system.check_observability())

