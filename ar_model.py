import numpy as np

class ARmodel:
    """Autoregressive model

    Args:
        r (int): autoregresion order
        decay (float, optional): Decay rate for exponential window, for rectangular
            window leaf default 1.0 value
    """
    def __init__(self, r, decay=1.0):
        assert decay>0 and decay<=1, 'Decay has to be in range (0, 1>'
        self.r = r
        self.decay = decay

        self.coefs = np.zeros((1,self.r), dtype=np.float)

    def updateParams(self, y):
        """Calculate new AR coefficients using weighted least square algorithm

        Args:
            y (np.array): Signal to calculate AR coefficients
        """
        y = np.flip(y)
        y = np.expand_dims(y, axis=1)

        R = np.zeros((self.r, self.r), dtype=np.float)
        s = np.zeros((self.r), dtype=np.float)
        for i in range(1, y.shape[0] - self.r - 1):
            w = self.decay**i
            i_end = i + self.r
            R += w * np.matmul(y[i:i_end], y[i:i_end].T)
            s += w * y[i-1] * y[i:i_end,0]

        self.coefs = np.matmul(np.linalg.inv(R), s)
        self.coefs = np.flip(self.coefs)

    def estimateSignal(self, n, x):
        """Estimate signal based on initial samples and current AR coefficients

        Args:
            n (int): number of samples to estimate
            x (np.array): initial samples
        Returns:
            np.array: estimated signal
        """
        y = np.zeros((self.r+n), dtype=np.float)
        y[:self.r] = x[-self.r:]

        for i in range(self.r, y.shape[0]):
            y[i] = np.matmul(y[i-self.r:i], self.coefs)

        return y[self.r:]
