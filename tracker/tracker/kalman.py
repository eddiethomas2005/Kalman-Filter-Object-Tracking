"""Constant velocity Kalman filter, NumPy only (no cv2.KalmanFilter). Knows nothing about video."""

import numpy as np


class KalmanFilter:
    """State is [px, py, vx, vy], measurement is just [px, py]."""

    def __init__(self, dt, process_var, meas_var, init_pos_var, init_vel_var):
        self.dt = dt
        self.init_pos_var = init_pos_var
        self.init_vel_var = init_vel_var

        self.F = np.eye(4) # F is the physics and prediction model we need to update the position by p + v*dt
        self.F[0, 2] = self.dt
        self.F[1,3] = self.dt

        # only ever measure position so H just picks px, py out
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)

        # Q: white noise acceleration, same for x and y.
        # state is [px, py, vx, vy] not [px, vx, py, vy] hence the ix_ indexing
        q = np.array([[dt**4 / 4, dt**3 / 2],
                      [dt**3 / 2, dt**2]])
        self.Q = np.zeros((4, 4))
        self.Q[np.ix_([0, 2], [0, 2])] = q   # x: px, vx
        self.Q[np.ix_([1, 3], [1, 3])] = q   # y: py, vy
        self.Q *= process_var

        self.R = meas_var * np.eye(2)

        self.x = np.zeros(4)
        self.P = np.eye(4)
        self.initialized = False

    def initialize(self, z):
        """Start the track on the first detection, velocity 0 with big uncertainty."""
        self.x= np.array([z[0],z[1],0,0] , dtype=float)
        self.P = self.init_pos_var * np.eye(4)
        self.P[2,2] = self.init_vel_var
        self.P[3,3] = self.init_vel_var
        self.initialized = True

    def predict(self):
        """x = Fx, P = FPF^T + Q."""
        self.x= self.F @ self.x
        self.P=self.F @ self.P @ np.transpose(self.F) + self.Q
        return self.x

        

    def update(self, z):
        """Correct the prediction with z.

        y = z - Hx, S = HPH^T + R, K = PH^T S^-1
        x = x + Ky, P = (I - KH)P
        """
        # solve instead of inv(S). K = (S^-1 H P)^T works since P and S are symmetric
        y=z- self.H @ self.x
        s = self.H @ self.P @ np.transpose(self.H) + self.R
        k =np.linalg.solve(s, self.H @ self.P).T
        self.x = self.x + (k @ y)
        self.P = (np.eye(4)- k@self.H) @ self.P 
        return self.x
        

    def innovation_distance(self, z):
        """y^T S^-1 y (squared Mahalanobis), the tracker uses this to reject bad detections."""
        y=z- self.H @ self.x
        s = self.H @ self.P @ np.transpose(self.H) + self.R
        return float(y @ np.linalg.solve(s, y))
        

    @property
    def position(self):
        return self.x[:2].copy()

    @property
    def position_cov(self):
        """Top left 2x2 of P, for the ellipse."""
        return self.P[:2, :2].copy()
