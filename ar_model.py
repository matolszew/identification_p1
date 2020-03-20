import numpy as np

class ARmodel:
    """

    Args:
        r (int): autoregresion order
        decay (float, optional): Decay rate for exponential window, for rectangular
            window leaf default 1.0 value
    """
    def __init__(self, r, decay=1.0):
        assert decay>0 and decay<=1, 'Decay has to be in range (0, 1>'
        self.r = r
        self.decay = decay

        self.coefs = np.zeros((self.r), dtype=np.float)

    def updateParams(self, y):
        """

        Args:
            y (np.array): Signal to calculate AR coefficients
        """
        y = np.flip(y)
        y = np.expand_dims(y, axis=1)

        R = np.zeros((self.r, self.r), dtype=np.float)
        s = np.zeros((self.r), dtype=np.float)

        for i in range(y.shape[0] - self.r - 1):
            w = self.decay**i
            i_end = i + self.r + 1
            R += w * np.matmul(y[i+1 : i_end], y[i+1:i_end].T)
            s += w * y[i] * y[i+1:i_end,0]

        self.coefs = np.matmul(np.linalg.inv(R), s)

    def estimateSignal(self):
        """
        """
