# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""An abstract objective function definition and common objective functions suitable
for classifiers/regressors."""

from abc import abstractmethod
from typing import Optional, Union

import numpy as np

try:
    from sparse import SparseArray
except ImportError:

    class SparseArray:  # type: ignore
        """Empty SparseArray class
        Replacement if sparse.SparseArray is not present.
        """

        pass


from qiskit_machine_learning.neural_networks import NeuralNetwork
from qiskit_machine_learning.utils.loss_functions import Loss

from qiskit.utils import algorithm_globals


class ObjectiveFunction:
    """An abstract objective function. Provides methods for computing objective value and
    gradients for forward and backward passes."""

    # pylint: disable=invalid-name
    def __init__(
        self, X: np.ndarray, y: np.ndarray, neural_network: NeuralNetwork, loss: Loss, batch_size: int = None
    ) -> None:
        """
        Args:
            X: The input data.
            y: The target values.
            neural_network: An instance of an quantum neural network to be used by this
                objective function.
            loss: A target loss function to be used in training.
        """
        super().__init__()
        self._X = X
        self._y = y
        self._neural_network = neural_network
        self._loss = loss
        self._last_forward_weights: Optional[np.ndarray] = None
        self._last_forward: Optional[Union[np.ndarray, SparseArray]] = None
        if batch_size is None:
            self._is_stochastic = False
        else:
            self._is_stochastic = True
            self._batch_size = batch_size


    @abstractmethod
    def objective(self, weights: np.ndarray, seed: int = None) -> float:
        """Computes the value of this objective function given weights.

        Args:
            weights: an array of weights to be used in the objective function.
            seed: random seed used to sample batch, if using SGD

        Returns:
            Value of the function.
        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Computes gradients of this objective function given weights.

        Args:
            weights: an array of weights to be used in the objective function.
            seed: random seed used to sample batch, if using SGD

        Returns:
            Gradients of the function.
        """
        if self._is_stochastic:
            print("Warning: Gradient method has not been adapted to stochastic gradient descent.")
        raise NotImplementedError

    def _neural_network_forward(self, weights: np.ndarray, seed: int = None) -> Union[np.ndarray, SparseArray]:
        """
        Computes and caches the results of the forward pass. Cached values may be re-used in
        gradient computation.

        Args:
            weights: an array of weights to be used in the forward pass.
            seed: random seed used to sample batch, if using SGD

        Returns:
            The result of the neural network.
        """
        # if we get the same weights, we don't compute the forward pass again.
        if self._last_forward_weights is None or (
            not np.all(np.isclose(weights, self._last_forward_weights))
        ):
            if self._is_stochastic:
                rng = np.random.default_rng(seed)
                samples = rng.integers(0, len(self._X), self._batch_size)
                X = self._X[samples]
            else:
                X = self._X
            # compute forward and cache the results for re-use in backward
            self._last_forward = self._neural_network.forward(X, weights)
            # a copy avoids keeping a reference to the same array, so we are sure we have
            # different arrays on the next iteration.
            self._last_forward_weights = np.copy(weights)
        return self._last_forward


class BinaryObjectiveFunction(ObjectiveFunction):
    """An objective function for binary representation of the output,
    e.g. classes of ``-1`` and ``+1``."""

    def objective(self, weights: np.ndarray, seed: int = None) -> float:
        # predict is of shape (N, 1), where N is a number of samples
        predict = self._neural_network_forward(weights, seed)
        if self._is_stochastic:
            rng = np.random.default_rng(seed)
            samples = rng.integers(0, len(self._X), self._batch_size)
            y = self._y[samples]
        else:
            y = self._y

        target = np.array(y).reshape(predict.shape)
        # float(...) is for mypy compliance
        return float(np.sum(self._loss(predict, target)))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        # check that we have supported output shape
        num_outputs = self._neural_network.output_shape[0]
        if num_outputs != 1:
            raise ValueError(f"Number of outputs is expected to be 1, got {num_outputs}")

        # output must be of shape (N, 1), where N is a number of samples
        output = self._neural_network_forward(weights)
        # weight grad is of shape (N, 1, num_weights)
        _, weight_grad = self._neural_network.backward(self._X, weights)

        # we reshape _y since the output has the shape (N, 1) and _y has (N,)
        # loss_gradient is of shape (N, 1)
        loss_gradient = self._loss.gradient(output, self._y.reshape(-1, 1))

        # for the output we compute a dot product(matmul) of loss gradient for this output
        # and weights for this output.
        grad = loss_gradient[:, 0] @ weight_grad[:, 0, :]
        # we keep the shape of (1, num_weights)
        grad = grad.reshape(1, -1)

        return grad


class MultiClassObjectiveFunction(ObjectiveFunction):
    """
    An objective function for multiclass representation of the output,
    e.g. classes of ``0``, ``1``, ``2``, etc.
    """

    def objective(self, weights: np.ndarray, seed: int = None) -> float:
        # probabilities is of shape (N, num_outputs)
        probs = self._neural_network_forward(weights, seed)

        num_outputs = self._neural_network.output_shape[0]
        val = 0.0
        if self._is_stochastic:
            num_samples = self._batch_size
            rng = np.random.default_rng(seed)
            samples = rng.integers(0, len(self._X), self._batch_size)
            y = self._y[samples]
        else:
            num_samples = self._X.shape[0]
            y = self._y
        for i in range(num_outputs):
            # for each output we compute a dot product of probabilities of this output and a loss
            # vector.
            # loss vector is a loss of a particular output value(value of i) versus true labels.
            # we do this across all samples.
            val += probs[:, i] @ self._loss(np.full(num_samples, i), y)

        return val

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        # weight probability gradient is of shape (N, num_outputs, num_weights)
        _, weight_prob_grad = self._neural_network.backward(self._X, weights)

        grad = np.zeros((1, self._neural_network.num_weights))
        num_samples = self._X.shape[0]
        num_outputs = self._neural_network.output_shape[0]
        for i in range(num_outputs):
            # similar to what is in the objective, but we compute a matrix multiplication of
            # weight probability gradients and a loss vector.
            grad += weight_prob_grad[:, i, :].T @ self._loss(np.full(num_samples, i), self._y)

        return grad


class OneHotObjectiveFunction(ObjectiveFunction):
    """
    An objective function for one hot encoding representation of the output,
    e.g. classes like ``[1, 0, 0]``, ``[0, 1, 0]``, ``[0, 0, 1]``.
    """

    def objective(self, weights: np.ndarray, seed: int = None) -> float:
        # probabilities is of shape (N, num_outputs)
        probs = self._neural_network_forward(weights, seed)
        if self._is_stochastic:
            rng = np.random.default_rng(seed)
            samples = rng.integers(0, len(self._X), self._batch_size)
            y = self._y[samples]
        else:
            y = self._y
        # float(...) is for mypy compliance
        return float(np.sum(self._loss(probs, y)))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        # predict is of shape (N, num_outputs)
        y_predict = self._neural_network_forward(weights)
        # weight probability gradient is of shape (N, num_outputs, num_weights)
        _, weight_prob_grad = self._neural_network.backward(self._X, weights)

        grad = np.zeros(self._neural_network.num_weights)
        num_outputs = self._neural_network.output_shape[0]
        # loss gradient is of shape (N, num_output)
        loss_gradient = self._loss.gradient(y_predict, self._y)
        for i in range(num_outputs):
            # a dot product(matmul) of loss gradient and weight probability gradient across all
            # samples for an output.
            grad += loss_gradient[:, i] @ weight_prob_grad[:, i, :]

        return grad
