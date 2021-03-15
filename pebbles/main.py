from pebbles import PebbleGame
import time

game = PebbleGame(3, (0,2))

#for i in [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]:
#    start = time.time()
#    game.simulate(i)
#    print(f'time: {time.time() - start}; iterations: {i}')

game.simulate(3476)
#game.plotBoard()
game.plotProbsBlock((1,1), True)

