import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Callable, Tuple, Optional

class DownhillSimplexAlgorithm:
    """
    Implementation of the Downhill Simplex (Nelder-Mead) optimization algorithm.

    Parameters:
    -----------
    func : Callable
        The objective function to be minimized.
    dim : int
        Dimensionality of the input space.
    initial_simplex : Optional[np.array], optional
        Initial simplex, a (dim + 1) x dim array. If None, a random simplex will be generated.
    alpha : float, optional
        Reflection coefficient. Default is 1.0.
    gamma : float, optional
        Expansion coefficient. Default is 1.5.
    rho : float, optional
        Contraction coefficient. Default is 0.5.
    sigma : float, optional
        Shrinkage coefficient. Default is 0.5.
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    """

    def __init__(self, 
                 func: Callable[[np.array], float], 
                 dim: int, 
                 initial_simplex: Optional[np.array] = None,
                 alpha: float = 1.0, 
                 gamma: float = 1.5, 
                 rho: float = 0.5, 
                 sigma: float = 0.5, 
                 tol: float = 1e-6, 
                 max_iter: int = 1000):
        self.func = func
        self.dim = dim
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.tol = tol
        self.max_iter = max_iter

        if initial_simplex is not None:
            self.simplex = initial_simplex
        else:
            self.simplex = np.random.randn(dim + 1, dim)

        self.history = []

    def _sort_simplex(self) -> None:
        """
        Sort the simplex vertices in ascending order based on their function values.
        """
        self.simplex = self.simplex[np.argsort([self.func(x) for x in self.simplex])]

    def _centroid(self) -> np.array:
        """
        Compute the centroid of the simplex excluding the worst point.

        Returns:
        --------
        np.array
            The centroid of the simplex.
        """
        return np.mean(self.simplex[:-1], axis=0)

    def reflect(self, x_o: np.array, x_w: np.array) -> np.array:
        """
        Perform reflection of the worst point.

        Parameters:
        -----------
        x_o : np.array
            The centroid of the simplex.
        x_w : np.array
            The worst point in the simplex.

        Returns:
        --------
        np.array
            The reflected point.
        """
        return x_o + self.alpha * (x_o - x_w)

    def expand(self, x_o: np.array, x_r: np.array) -> np.array:
        """
        Perform expansion of the reflected point.

        Parameters:
        -----------
        x_o : np.array
            The centroid of the simplex.
        x_r : np.array
            The reflected point.

        Returns:
        --------
        np.array
            The expanded point.
        """
        return x_o + self.gamma * (x_r - x_o)

    def contract(self, x_o: np.array, x_r: np.array, outside: bool = True) -> np.array:
        """
        Perform contraction.

        Parameters:
        -----------
        x_o : np.array
            The centroid of the simplex.
        x_r : np.array
            The reflected point.
        outside : bool, optional
            Flag indicating whether to perform outside or inside contraction. Default is True (outside).

        Returns:
        --------
        np.array
            The contracted point.
        """
        if outside:
            return x_o + self.rho * (x_r - x_o)
        else:
            return x_o + self.rho * (self.simplex[-1] - x_o)

    def shrink(self) -> None:
        """
        Shrink the simplex towards the best point.
        """
        x_best = self.simplex[0]
        self.simplex = x_best + self.sigma * (self.simplex - x_best)

    def optimize(self) -> Tuple[np.array, float, list]:
        """
        Optimize the objective function using the Downhill Simplex algorithm.

        Returns:
        --------
        Tuple[np.array, float, list]
            The best point found, its corresponding function value, and the history of best points.
        """
        for _ in range(self.max_iter):
            self._sort_simplex()
            
            f_min = self.func(self.simplex[0])
            f_max = self.func(self.simplex[-1])
            if f_max - f_min < self.tol:
                break

            x_o = self._centroid()

            x_r = self.reflect(x_o, self.simplex[-1])
            f_r = self.func(x_r)
            if f_min <= f_r < self.func(self.simplex[-2]):
                self.simplex[-1] = x_r
            elif f_r < f_min:
                x_e = self.expand(x_o, x_r)
                if self.func(x_e) < self.func(x_r):
                    self.simplex[-1] = x_e
                else:
                    self.simplex[-1] = x_r
            else:
                if f_r < f_max:
                    x_c = self.contract(x_o, x_r, outside=True)
                else:
                    x_c = self.contract(x_o, self.simplex[-1], outside=False)
            
                if self.func(x_c) < min(f_r, f_max):
                    self.simplex[-1] = x_c
                else:
                    self.shrink()

            self.history.append(self.simplex[0].copy())

        self._sort_simplex()
        return self.simplex[0], self.func(self.simplex[0]), self.history

def Rosenbrock_function(point: np.array, a: float = 1, b: float = 100) -> float:
    """
    Compute the Rosenbrock function.

    Parameters:
    -----------
    point : np.array
        The input point (x, y).
    a : float, optional
        Parameter a of the Rosenbrock function. Default is 1.
    b : float, optional
        Parameter b of the Rosenbrock function. Default is 100.

    Returns:
    --------
    float
        The value of the Rosenbrock function at the given point.
    """
    x, y = point
    return (a - x) ** 2 + b * (y - x ** 2) ** 2

def plot_trajectory_with_heatmap(history: list, func: Callable) -> None:
    history = np.array(history)

    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func([x_i, y_i]) for x_i, y_i in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])
    
    fig, ax = plt.subplots()
    ax.set_title("Optimization Trajectory with Heatmap")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    heatmap = ax.imshow(Z, extent=(-2, 2, -1, 3), origin='lower', cmap='hot', alpha=0.6)

    point, = ax.plot([], [], 'bo-', label="Trajectory")
    final_point, = ax.plot([], [], 'ro', label='Final Best Point')

    def init():
        point.set_data([], [])
        final_point.set_data([], [])
        return point, final_point,

    def animate(i):
        point.set_data(history[:i+1, 0], history[:i+1, 1])
        final_point.set_data([history[i, 0]], [history[i, 1]])
        return point, final_point,

    ani = animation.FuncAnimation(fig, animate, frames=len(history), init_func=init, blit=True, repeat=False)

    ax.legend()
    plt.colorbar(heatmap, ax=ax, label='Function Value')
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    dsa = DownhillSimplexAlgorithm(Rosenbrock_function, dim=2)
    best_point, best_val, history = dsa.optimize()
    print(f"Best Point: {best_point}, Best Value: {best_val}")
    plot_trajectory_with_heatmap(history, Rosenbrock_function)