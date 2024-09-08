import os  
import matplotlib.pyplot as plt  
import numpy as np  

class Logger:  
    """  
    A simple logger class to track and plot values over iterations.  
    """  

    def __init__(self, name: str, path: str = 'log') -> None:  

        os.makedirs(path, exist_ok=True)  
        self._value_curve = []  
        self._iterations = []  
        self.name = name  
        self.path = path  

    def add_value(self, loss: float) -> None:  

        self._value_curve.append(loss)  
        self._iterations.append(len(self._value_curve))  

    def plot(self) -> None:  
        plt.clf()  

        if self._value_curve:  
            iterations = np.array(self._iterations)  
            plt.plot(iterations, np.array(self._value_curve), '-r')  

        plt.title(f'{self.name} Curve')  
        plt.xlabel('Iteration')  
        plt.ylabel(self.name)  

        plot_path = os.path.join(self.path, f'{self.name}.png')  
        plt.savefig(plot_path)  