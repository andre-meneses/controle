import control as ctrl
import numpy as np
import sympy as sp

class ContinousSystemAnalysis:
    def __init__(self, A, B, C, D, Ts=None):
        self.Ts = Ts

        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.sys = ctrl.StateSpace(A, B, C, D)

    def check_stability(self, print_poly=False):
        """
        Checks the stability of the continuous-time state-space model.

        Parameters:
        print_poly (bool): If True, prints the characteristic polynomial of the system matrix.
        """
        A_continuous = self.A  # Make sure to use the continuous-time system matrix
        eigenvalues = np.linalg.eigvals(A_continuous)
        print("Eigenvalues of the system matrix A:")
        sp.pprint(sp.Matrix(eigenvalues))

        # Print the characteristic polynomial if requested
        if print_poly:
            s = sp.symbols('s')
            # Convert the numpy matrix A_continuous to a sympy Matrix if it's not already one
            if not isinstance(A_continuous, sp.Matrix):
                A_continuous = sp.Matrix(A_continuous)
            char_poly = A_continuous.charpoly(s)
            print("Characteristic polynomial of the system matrix A:")
            sp.pprint(char_poly.as_expr())

        # A continuous-time system is stable if all eigenvalues have negative real parts
        is_stable = np.all(np.real(eigenvalues) < 0)
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

    def ackermann_state_feedback(self, desired_poles, augment=False):
        """
        Computes the state feedback gain matrix K using Ackermann's formula.
        """
        if augment:
            A = self.A_aug
            B = self.B_aug
        else:
            A = self.A
            B = self.B

        cont_matrix = ctrl.ctrb(A, B)

        if np.linalg.matrix_rank(cont_matrix) != A.shape[0]:
            raise ValueError("The system is not controllable and pole placement cannot be performed.")

        K = ctrl.acker(A, B, desired_poles)

        K = -1*K

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
        A = self.A
        B = self.B
        C = self.C
        # Original system dimensions

        n = A.shape[0]  # Number of states
        m = B.shape[1]  # Number of inputs
        p = C.shape[0]  # Number of outputs

        A_aug = np.block([
            [np.zeros((p, p)), C],
            [np.zeros((n, p)), A]
        ])
        B_aug = np.vstack((np.zeros((p, m)), B))

        self.A_aug = A_aug
        self.B_aug = B_aug


    def reference_tracker_gain(self, desired_poles):

        k = self.ackermann_state_feedback(desired_poles, augment=True)
        k1 = k[0,0]
        k2 = k[0,1:]

        return k1,k2

    def char_poly(self):
        s = sp.symbols('s')
        return sp.Matrix(self.A).charpoly(s).as_expr()

if __name__ == '__main__':
    A = np.array([[0, 1], [0, -2]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0], [0, 0]])
    D = np.array([[0], [0]])
    Ts = 1  # Sampling time

    # Example for discrete system
    system = ContinousSystemAnalysis(A, B, C, D, Ts)
    print("Is the system stable?", system.check_stability())
    print("Is the system controllable?", system.check_controllability())
    print("Is the system observable?", system.check_observability())

    # For already discrete systems, set Ts=None
    discrete_system = ContinousSystemAnalysis(A, B, C, D)
    print("Is the discrete system stable?", discrete_system.check_stability())
    print("Is the discrete system controllable?", discrete_system.check_controllability())
    print("Is the discrete system observable?", discrete_system.check_observability())

