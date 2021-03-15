import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


class PebbleGame ():
    #plt.style.use('ggplot')

    def __init__(self, board_size, initial_position):
        self.board = np.zeros((board_size, board_size))

        assert (initial_position[0] >= 0 and initial_position[0] < board_size) and (initial_position[1] >= 0 and initial_position[1] < board_size), 'invalid positions!'
    
        self.i = initial_position[0]
        self.j = initial_position[1]

    def simulate(self, total_iterations):
        self.total_iterations = total_iterations
        self.board, self.board_per_iteration = self._simulate(self.board, total_iterations, self.i, self.j)
        return self.board, self.board_per_iteration
        
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

    def plotProbs(self):
        plt.figure(figsize=(7,7), dpi=200)

        # iterate over positions
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                position = []

                for num, board in enumerate(self.board_per_iteration):
                    position.append(board[i,j]/np.sum(board))

                plt.plot(range(self.total_iterations), position, marker='', label=f'{i},{j}')
                
        plt.legend()
        plt.show()

    def plotProbsBlock(self, p, zoom):
        fig, ax = plt.subplots(figsize=(7,7), dpi=200)

        # iterate over positions 4 hidden plots
        for i in range(len(self.board)):
            for j in range(len(self.board)):
        
                position = []
                for board in self.board_per_iteration:
                    position.append(board[i,j]/np.sum(board))

                ax.plot(range(self.total_iterations), position, marker='', linestyle="--", color='grey', linewidth=1., alpha=0.3)

        # real plot
        x, y = p
        position = []
        for board in self.board_per_iteration:
            position.append(board[x, y]/np.sum(board))

        ax.plot(range(self.total_iterations), position, linewidth=2, color='r', label=f'({i}, {j})')
             
        # analytic prob
        a = np.full((self.total_iterations), 1/9)
        ax.plot(range(self.total_iterations), a, 'k-', linestyle = "-", lw=1.5)

        # configs
        plt.xlim(0,self.total_iterations)
        plt.ylim(0,1)
        plt.yticks(np.arange(0,1.05,1/9), [r'$0$',r'$\frac{1}{9}$',r'$\frac{2}{9}$', r'$\frac{3}{9}$',r'$\frac{4}{9}$',r'$\frac{5}{9}$',r'$\frac{6}{9}$',r'$\frac{7}{9}$',r'$\frac{8}{9}$', r'$1$'])

        # zoom 
        if zoom == True:
            axins = zoomed_inset_axes(ax, 6, loc=1) # zoom = 6
            axins.plot(range(self.total_iterations), position, linewidth=2, color='r', label=f'({i}, {j})')
            axins.plot(range(self.total_iterations), a, 'k-', linestyle = "-", lw=1.5)
            axins.set_xlim(self.total_iterations-(self.total_iterations*.1), self.total_iterations) # Limit the region for zoom
            axins.set_ylim((1/9-0.05), (1/9+0.05))

            plt.xticks(visible=False)  # Not present ticks
            #plt.yticks(visible=False)
            #
            ## draw a bbox of the region of the inset axes in the parent axes and
            ## connecting lines between the bbox and the inset axes area
            mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec=".5")


        # general title
        ax.set_title(f'Probabilities Over Time of {p}')
        plt.draw()
        plt.show()


    @staticmethod
    @jit(nopython=True) # use numba
    def _simulate(board, total_iterations, i, j):
        #probs_over_time = []
        board_per_iteration = np.empty((total_iterations, len(board), len(board)))

        board_per_iteration[0] = board

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

            board_per_iteration[_] = board

        return board, board_per_iteration
