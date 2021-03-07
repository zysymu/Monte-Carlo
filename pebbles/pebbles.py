import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit

class PebbleGame ():
    def __init__(self, board_size, initial_position):
        self.board = np.zeros((board_size, board_size))

        assert (initial_position[0] >= 0 and initial_position[0] < board_size) and (initial_position[1] >= 0 and initial_position[1] < board_size), 'invalid positions!'
    
        self.i = initial_position[0]
        self.j = initial_position[1]

    def simulate(self, total_iterations):
        self.total_iterations = total_iterations
        self.board = self._simulate_helper(self.board, total_iterations, self.i, self.j)
        
    def printBoard(self):
        print(self.board)

    def plotBoard(self):
        plt.figure(figsize=(7,7), dpi=200)

        group_counts = ['{0:0.0f}'.format(value) for value in self.board.flatten()]
        group_percentages = ['({0:.2%})'.format(value) for value in self.board.flatten()/self.total_iterations]
        labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(self.board.shape)

        sns.heatmap(self.board, annot=labels, fmt="", linewidths=.5, cmap="PiYG")
        plt.title(f'Markov-chain Discrete Pebble Game Distribution over {int(self.total_iterations)} iterations')
        plt.show()

    @staticmethod
    @jit(nopython=True) # use numba
    def _simulate_helper(board, total_iterations, i, j):
        # generate random numbers
        for _ in range(total_iterations):
            board[i,j] += 1 # counts the initial position
           
            r = np.random.rand()

            # make movement:
            if r < 0.25: #up
                aux = j+1
                j=aux if (aux >= 0 and aux < len(board)) else j

            if 0.25 <= r < 0.5: # left
                aux = i-1
                i=aux if (aux >= 0 and aux < len(board)) else i

            if 0.5 <= r < 0.75: # down
                aux = j-1
                j=aux if (aux >= 0 and aux < len(board)) else j

            if 0.75 <= r < 1: # right
                aux = i+1
                i=aux if (aux >= 0 and aux < len(board)) else i

        return board
