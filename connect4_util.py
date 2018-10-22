import numpy as np

brd_sh = [6, 7]
act_sz = brd_sh[1]

def to_one_hot(brd):
    #print(brd)
    tmp = np.eye(3)[brd]
    #print(tmp.shape)
    return np.transpose(np.eye(3)[brd], [0, 3, 1, 2])

def over(brd):
    for i in range(brd_sh[0]):
        prev = 0
        cnt = 0
        line = brd[i]
        for val in line:
            if val != 0:
                if val == prev:
                    cnt += 1
                    if cnt == 4:
                        return val
                else:
                    cnt = 1
            else:
                cnt = 0
            prev = val
            
    for i in range(brd_sh[1]):
        prev = 0
        cnt = 0
        line = brd.T[i]
        for val in line:
            if val != 0:
                if val == prev:
                    cnt += 1
                    if cnt == 4:
                        return val
                else:
                    cnt = 1
            else:
                cnt = 0
            prev = val
    
    for i in range(-brd_sh[0]+1, brd_sh[1]):
        prev = 0
        cnt = 0
        line = np.diag(brd, i)
        for val in line:
            if val != 0:
                if val == prev:
                    cnt += 1
                    if cnt == 4:
                        return val
                else:
                    cnt = 1
            else:
                cnt = 0
            prev = val
    for i in range(-brd_sh[0]+1, brd_sh[1]):
        prev = 0
        cnt = 0
        line = np.diag(np.fliplr(brd), i)
        for val in line:
            if val != 0:
                if val == prev:
                    cnt += 1
                    if cnt == 4:
                        return val
                else:
                    cnt = 1
            else:
                cnt = 0
            prev = val
    
    if np.any(brd == 0):
        return 0
    return -1

def foul(brd, act):
    return brd[0, act] != 0

def place(brd, act, player):
    for i in range(brd_sh[0]-1):
        if brd[i+1][act] != 0:
            brd[i][act] = player
            return
    brd[brd_sh[0]-1, act] = player

def flip(brd):
    brd[brd == 1] = 3
    brd[brd == 2] = 1
    brd[brd == 3] = 2
    #return np.select([brd == 1, brd == 2], [2, 1])
