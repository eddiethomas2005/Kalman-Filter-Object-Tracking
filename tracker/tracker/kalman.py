"""Constant-velocity Kalman filter in pure NumPy. Knows nothing about video."""

import numpy as np


class KalmanFilter:
    """State x = [px, py, vx, vy]^T, measurement z = [px, py]^T."""

    def __init__(self, dt, process_var, meas_var, init_pos_var, init_vel_var):
        self.dt = dt
        self.init_pos_var = init_pos_var
        self.init_vel_var = init_vel_var

        # TODO: state transition. Position advances by velocity * dt.
        self.F = np.eye(4) # F is the physics and prediction model we need to update the position by p + v*dt
        self.F[0, 2] = self.dt
        self.F[1,3] = self.dt

        # Measurement model: we observe position only.
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)

        # Process noise: discrete white-noise acceleration, independent per axis.
        # State order is [px, py, vx, vy], so each axis's 2x2 block is spread out.
        q = np.array([[dt**4 / 4, dt**3 / 2],
                      [dt**3 / 2, dt**2]])
        self.Q = np.zeros((4, 4))
        self.Q[np.ix_([0, 2], [0, 2])] = q   # x axis: px, vx
        self.Q[np.ix_([1, 3], [1, 3])] = q   # y axis: py, vy
        self.Q *= process_var

        self.R = meas_var * np.eye(2)

        self.x = np.zeros(4)
        self.P = np.eye(4)
        self.initialized = False

    def initialize(self, z):
        """Start the track at the first detection with zero velocity."""
        self.x= np.array([z[0],z[1],0,0] , dtype=float)
        self.P = self.init_pos_var * self.P
        self.P[2,2] = self.init_vel_var
        self.P[3,3] = self.init_vel_var
        self.initialized = True

    def predict(self):
        """x = F x ; P = F P F^T + Q. Returns the predicted state."""
        self.x= self.F @ self.x
        self.P=self.F @ self.P @ np.transpose(self.F) + self.Q
        return self.x

        

    def update(self, z):
        """Correct the prediction with measurement z. Returns the updated state.

        y = z - H x                  (innovation)
        S = H P H^T + R              (innovation covariance)
        K = P H^T S^-1               (Kalman gain)
        x = x + K y
        P = (I - K H) P
        """
        # TODO. Tip: np.linalg.solve is more stable than inverting S directly.
        y=z- self.H @ self.x
        s = self.H @ self.P @ np.transpose(self.H) + self.R
        k =np.linalg.solve(s, self.H @ self.P).T
        self.x = self.x + (k @ y)
        self.P = (np.eye(4)- k@self.H) @ self.P 
        return self.x
        

    def innovation_distance(self, z):
        """Squared Mahalanobis distance y^T S^-1 y, used to gate outlier detections."""
        # TODO
        y=z- self.H @ self.x
        s = self.H @ self.P @ np.transpose(self.H) + self.R
        return float(y @ np.linalg.solve(s, y))
        

    @property
    def position(self):
        return self.x[:2].copy()

    @property
    def position_cov(self):
        """2x2 position covariance, for drawing the uncertainty ellipse."""
        return self.P[:2, :2].copy()
