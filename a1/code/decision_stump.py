import numpy as np
import utils


class DecisionStumpEquality:
    """
    This is a decision stump that branches on whether the value of X is
    "almost equal to" some threshold.

    This probably isn't a thing you want to actually do, it's just an example.
    """

    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                # Choose value to equate to
                t = np.round(X[i, j])

                # Find most likely class for each split
                is_almost_equal = np.round(X[:, j]) == t
                y_yes_mode = utils.mode(y[is_almost_equal])
                y_no_mode = utils.mode(y[~is_almost_equal])  # ~ is "logical not"

                # Make predictions
                y_pred = y_yes_mode * np.ones(n)
                y_pred[np.round(X[:, j]) != t] = y_no_mode

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode

    def predict(self, X):
        n, d = X.shape
        X = np.round(X)

        if self.j_best is None:
            return self.y_hat_yes * np.ones(n)

        y_hat = np.zeros(n)

        for i in range(n):
            if X[i, self.j_best] == self.t_best:
                y_hat[i] = self.y_hat_yes
            else:
                y_hat[i] = self.y_hat_no

        return y_hat


class DecisionStumpErrorRate:
    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape
        
        count = np.bincount(y)

        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        for j in range(d):
            for i in range(n):
                t = np.round(X[i, j])

                # Find most likely class for each split
                is_greater =X[:, j] > t
                y_yes_mode = utils.mode(y[is_greater])
                y_no_mode = utils.mode(y[~is_greater])

                # Make predictions
                y_pred = y_no_mode * np.ones(n)
                y_pred[is_greater] = y_yes_mode

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode
        

    def predict(self, X):
        n, d = X.shape

        if self.j_best is None:
            return self.y_hat_yes * np.ones(n)
        
        y_hat = np.zeros(n)

        for i in range(n):
            if X[i, self.j_best] > self.t_best:
                y_hat[i] = self.y_hat_yes
            else:
                y_hat[i] = self.y_hat_no

        return y_hat


def entropy(p):
    """
    A helper function that computes the entropy of the
    discrete distribution p (stored in a 1D numpy array).
    The elements of p should add up to 1.
    This function ensures lim p-->0 of p log(p) = 0
    which is mathematically true, but numerically results in NaN
    because log(0) returns -Inf.
    """
    plogp = 0 * p  # initialize full of zeros
    plogp[p > 0] = p[p > 0] * np.log(p[p > 0])  # only do the computation when p>0
    return -np.sum(plogp)


class DecisionStumpInfoGain(DecisionStumpErrorRate):
    # This is not required, but one way to simplify the code is
    # to have this class inherit from DecisionStumpErrorRate.
    # Which methods (init, fit, predict) do you need to overwrite?
    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape
        num_classes = np.unique(y).size

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y, minlength=num_classes)
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        # Entropy of the labels before splitting
        p = count / n
        entropy_before = entropy(p)

        maxGain = 0

        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                t = X[i, j]

                is_greater = X[:, j] > t
                n_yes = np.sum(is_greater)
                n_no = n - n_yes

                # Skip splits that don't actually separate the data
                if n_yes == 0 or n_no == 0:
                    continue

                y_yes = y[is_greater]
                y_no = y[~is_greater]

                count_yes = np.bincount(y_yes, minlength=num_classes)
                count_no = np.bincount(y_no, minlength=num_classes)

                p_yes = count_yes / n_yes
                p_no = count_no / n_no

                entropy_yes = entropy(p_yes)
                entropy_no = entropy(p_no)

                # Weighted entropy after the split
                entropy_after = (n_yes / n) * entropy_yes + (n_no / n) * entropy_no

                info_gain = entropy_before - entropy_after

                # Compare to the best gain so far
                if info_gain > maxGain:
                    maxGain = info_gain
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = utils.mode(y_yes)
                    self.y_hat_no = utils.mode(y_no)

                    
def predict(X):
    n, d = X.shape
    y_hat = np.zeros(n)

    for i in range(n):
        if X[i, 1] > -80.305106:  # top split: feature = latitude
            if X[i, 0] > 36.453576:  # submodel_yes split
                y_hat[i] = 0  # y_hat_yes of submodel_yes
            else:
                y_hat[i] = 1  # y_hat_no of submodel_yes
        else:
            if X[i, 0] > 37.669007:  # submodel_no split
                y_hat[i] = 0  # y_hat_yes of submodel_no
            else:
                y_hat[i] = 1  # y_hat_no of submodel_no

    return y_hat
