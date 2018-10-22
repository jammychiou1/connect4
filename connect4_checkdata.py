import numpy as np
arrays = np.load('training_data.npz')
brds = arrays['brds']
players = arrays['players']
ps = arrays['ps']
vs = arrays['vs']
while True:
    try:
        idx = int(input())
        print(brds[idx])
        print(players[idx])
        print(ps[idx])
        print(vs[idx])
    except ValueError:
        pass
