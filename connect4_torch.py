import torch as tc
import numpy as np
from connect4_util import *
import threading
import os
import queue

class GpuWorker(threading.Thread):
    def __init__(self):
        super().__init__()
        self.Q = queue.Queue()
        self.stop_request = threading.Event()

    def run(self):
        while not self.stop_request.isSet():
            brds = []
            players = []
            qs = []
            for i in range(32):
                try:
                    front = self.Q.get(False)
                    #front = self.Q.get(True, 0.001)
                    brds.append(front[0])
                    players.append(front[1])
                    qs.append(front[2])
                except queue.Empty:
                    break
            #print('batch size: {}'.format(len(brds)))
            if len(brds) > 0:
                net_p, net_v = model(np.array(brds), np.array(players))
                p = net_p.detach().cpu().numpy()
                v = net_v.detach().cpu().numpy()
                #print(v.shape)
                for i, q in enumerate(qs):
                    q.put((p[i], v[i]))
            
    def join(self, timeout=None):
        self.stop_request.set()
        super().join(timeout)

def evaluate(brd, player):
    Q = queue.Queue(1)
    gpu_worker.Q.put((brd, player, Q))
    p, v = Q.get()
    return p, v

MSE=tc.nn.MSELoss()
NLL=tc.nn.PoissonNLLLoss(log_input = False)
def train():
    l = 0
    for i in range(16):
        smpl = np.random.choice(replay_sz, smpl_sz)
        net_p, net_v = model(brds[smpl], players[smpl])
        optimizer.zero_grad()
        loss = MSE(net_v, tc.cuda.FloatTensor(vs[smpl])) + NLL(net_p, tc.cuda.FloatTensor(ps[smpl]))
        l += loss.item()
        loss.backward()
        optimizer.step()
    print('loss: {}\n'.format(l/16))

class SeachWorker(threading.Thread):
    def __init__(self, brd_beg, tree, player):
        super().__init__()
        self.brd_beg = brd_beg
        self.player = player
        self.tree = tree
        #self.lock = lock
        
    def run(self):
        c = 5
        for sim in range(10):
            brd1 = self.brd_beg
            nd1 = [self.tree]
            act1 = []
            while True:
                act1.append(np.argmax(nd1[-1].Q + c * nd1[-1].P * np.sqrt(nd1[-1].t) / (nd1[-1].T+1)))
                if foul(brd1, act1[-1]):
                    rslt = -5
                    break
                brd2 = np.copy(brd1)
                place(brd2, act1[-1], self.player)
                ov = over(brd2)
                if ov == self.player:
                    rslt = 1
                    break
                if ov == -1:
                    rslt = 0
                    break
                nd1[-1].lock.acquire()
                if nd1[-1].son[act1[-1]] is None:                
                    nd1[-1].son[act1[-1]] = Node2(brd2, self.player)
                nd1[-1].lock.release()
                nd2 = nd1[-1].son[act1[-1]]
                act2 = np.random.choice(act_sz, p = nd2.prob)
                if foul(brd2, act2):
                    rslt = 1
                    break
                place(brd2, act2, 3-self.player)
                ov = over(brd2)
                if ov == 3-self.player:
                    rslt = -1
                    break
                if ov == -1:
                    rslt = 0
                    break
                nd2.lock.acquire()
                if nd2.son[act2] is None:
                    nd2.son[act2] = Node1(brd2, self.player)
                    nd2.lock.release()
                    rslt = nd2.son[act2].q
                    #print(rslt)
                    break
                nd2.lock.release()
                brd1 = brd2
                nd1.append(nd2.son[act2])
            #self.lock.acquire()
            for nd, act in zip(nd1, act1):
                nd.lock.acquire()
                nd.Q[act] = (nd.Q[act] * nd.T[act] + rslt) / (nd.T[act] + 1)
                nd.T[act] += 1
                nd.t += 1
                nd.lock.release()
            #self.lock.release()

def mct(brd_beg, tree, player):
    #print('start mct')i
    #lock = threading.Lock()
    if tree is None:
        tree = Node1(brd_beg, player)
    seachers = [SeachWorker(brd_beg, tree, player) for i in range(80)]
    for seacher in seachers:
        seacher.start()
    for seacher in seachers:
        seacher.join()
    #print(tree.T.sum(), tree.t) 
    policy = tree.T / tree.t
    value = (policy * tree.Q).sum()
    return policy, value, tree

class Node1():
    def __init__(self, brd, player):
        self.Q = np.zeros(act_sz)
        self.T = np.zeros(act_sz)
        self.P, self.q = evaluate(brd, player)
        self.t = 0
        self.son = [None] * act_sz
        self.lock = threading.Lock()
        
class Node2():
    def __init__(self, brd, player):
        self.prob, _ = evaluate(brd, 3-player)
        self.son = [None] * act_sz
        self.lock = threading.Lock()

class Net(tc.nn.Module):    
    def __init__(self):
        super().__init__()  
        self.conv1 = tc.nn.Conv2d(4, 32, 2)
        self.conv2 = tc.nn.Conv2d(32, 64, 2)
        self.conv3 = tc.nn.Conv2d(64, 64, 2)
        self.fc1 = tc.nn.Linear(3*4*64, 64)
        self.fc2 = tc.nn.Linear(64, 32)
        self.fc3_p = tc.nn.Linear(32, 7)
        self.fc3_v = tc.nn.Linear(32, 1)
    def forward(self, brds, players):
        brds = tc.cuda.FloatTensor(to_one_hot(brds))
        players = tc.cuda.FloatTensor(players).view(-1, 1, 1, 1).expand(-1, 1, 6, 7)
        #print(brds.shape)
        #print(players.shape)
        net_in = tc.cat([brds, players], 1)
        #print(net_in.shape)
        net = tc.nn.functional.relu(self.conv1(net_in))
        net = tc.nn.functional.relu(self.conv2(net))
        net = tc.nn.functional.relu(self.conv3(net))
        net = net.view(-1, 3*4*64)
        net = tc.nn.functional.relu(self.fc1(net))
        net = tc.nn.functional.relu(self.fc2(net))
        net_p = tc.nn.functional.softmax(self.fc3_p(net), 1)
        #print(net_p)
        net_v = self.fc3_v(net).view(-1)
        #print(net_v)
        return net_p, net_v

def print_brd(brd):
    print('+---------------+')
    for i in range(brd_sh[0]):
        print('|', end=' ')
        for j in range(brd_sh[1]):
            if brd[i, j] == 0:
                print(' ', end=' ')
            if brd[i, j] == 1:
                print('O', end=' ')
            if brd[i, j] == 2:
                print('X', end=' ')
        print('|')
    print('+---------------+')
    print('  0 1 2 3 4 5 6 ')

if os.path.isfile('torch_model.pt'):
    print('load model')
    model = tc.load('torch_model.pt')
else:
    model = Net()
model.cuda()

optimizer = tc.optim.SGD(model.parameters(), lr = 0.01)

replay_sz = 10000
idx = 0
smpl_sz = 16

brds = np.zeros([replay_sz] + brd_sh, np.int32)
players = np.zeros(replay_sz, np.int32)
ps = np.zeros([replay_sz, act_sz])
vs = np.zeros(replay_sz)

gpu_worker = GpuWorker()
gpu_worker.start()

if os.path.isfile('training_data.npz'):
    arrays = np.load('training_data.npz')
    #print(arrays.files)
    brds = arrays['brds']
    players = arrays['players']
    ps = arrays['ps']
    vs = arrays['vs']             
else:
    print('preparing data')
    for step in range(replay_sz):
        print('step: {}'.format(step))
        brd = np.zeros(brd_sh, dtype=np.int32)
        act = 0
        player1 = 1
        player2 = 2
        tree1 = None
        tree2 = None
        while True:
            #print(tree1)
            #print(tree2)
            policy, value, tree1 = mct(brd, tree1, player1)
            brds[idx], players[idx], ps[idx], vs[idx] = brd, player1, policy, value
            idx = (idx + 1) % replay_sz
            act = np.random.choice(act_sz, p = policy)
            print(tree1.T)
            print(tree1.Q)
            print('policy:', policy)
            print('player {} plays {}'.format(player1, act))
            if foul(brd, act):
                #print('foul')
                break
            place(brd, act, player1)            
            print_brd(brd)
            if over(brd) == player1:
                print('player {} wins'.format(player1))
                break
            if over(brd) == -1:
                print('tie')
                break
            player1, player2 = player2, player1
            tree1 = tree1.son[act]
            if tree2 is not None:
                tree2 = tree2.son[act]
            tree1, tree2 = tree2, tree1
    np.savez('training_data.npz', brds=brds, players=players, ps=ps, vs=vs)

print('main training')
idx = np.random.randint(replay_sz)
for step in range(100000):
    print('step: {}'.format(step))
    brd = np.zeros(brd_sh, dtype=np.int32)
    act = 0
    player1 = 1
    player2 = 2
    tree1 = None
    tree2 = None
    while True:
        #if tree1 is not None:
        #    print(tree1.T)
        #    print(tree1.Q)
        policy, value, tree1 = mct(brd, tree1, player1)
        brds[idx], players[idx], ps[idx], vs[idx] = brd, player1, policy, value
        idx = (idx + 1) % replay_sz
        act = np.random.choice(act_sz, p = policy)
        print('policy:', policy)
        print('prev_pol:', tree1.P)
        print('value:', value)
        print('prev_val:', tree1.q)
        print('T:', tree1.T)
        print()
        print('player {} plays {}'.format(player1, act))
        if foul(brd, act):
            print('foul')
            break
        place(brd, act, player1)
        print_brd(brd)
        if over(brd) == player1:
            print('player {} wins'.format(player1))
            break
        if over(brd) == -1:
            print('tie')
            break
        print()
        player1, player2 = player2, player1
        tree1 = tree1.son[act]
        if tree2 is not None:
            tree2 = tree2.son[act]
        tree1, tree2 = tree2, tree1
        train()
    print('saving')
    tc.save(model, 'torch_model.pt')
    np.savez('training_data.npz', brds=brds, players=players, ps=ps, vs=vs)
