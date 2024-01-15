import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0
        self.total_size = 0
    
    def add(self, transition):
        assert len(transition) == 7, "transition must have length = 7"
        
        # transiton is tuple of (state, action, reward, next_state, goal, gamma, done)
        self.buffer.append(transition)
        self.size +=1
        self.total_size += 1 
        # Here a overflow protection is needed
        # if self.total_size >= 10e10:
        #   self.total_size = 0
    
    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size/5)]
            self.size = len(self.buffer)

        # print(f'\nlen(self.buffer) = {len(self.buffer)}\n')
        # print(f'\nbatch_size = {batch_size}\n')
        
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        states, actions, rewards, next_states, goals, gamma, dones = [], [], [], [], [], [], []
        
        for i in indexes:
            states.append(np.array(self.buffer[i][0], copy=False))
            actions.append(np.array(self.buffer[i][1], copy=False))
            rewards.append(np.array(self.buffer[i][2], copy=False))
            next_states.append(np.array(self.buffer[i][3], copy=False))
            goals.append(np.array(self.buffer[i][4], copy=False))
            gamma.append(np.array(self.buffer[i][5], copy=False))
            dones.append(np.array(self.buffer[i][6], copy=False))

        
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(goals),  np.array(gamma), np.array(dones)


def euclidean_distance(x, y):
    """Calculate the Euclidean distance between two NumPy arrays"""
    return np.sqrt(np.sum((x - y) ** 2))


# This is for plotting
class WindowMean(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.queue = deque()
        self.window_size = 10

    def calculate_window_average(self, number):
        self.queue.append(number)
        if len(self.queue) > self.window_size:
            self.queue.popleft()
        return sum(self.queue) / len(self.queue)
