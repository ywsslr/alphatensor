## A program to simulate the process of matric decomposition at sampled actions
## Just considerring the Strassen first  (finished)
import numpy as np
import random
from SampleActions import generate_unique_samples

# set the random seed 
seed = 42
np.random.seed(seed)
random.seed(seed)

class Decompositor:
    def __init__(self, num_samples:int=100):
        self.n = 2  # T_{m,n,p} with m=n=p=2
        self.size = self.n * self.n  # the size of T2
        self.F = (-1,0,1)  # the only element to be used
        self.gamma = 1  # upper bound
        self.R_limit = self.n ** 3 + 2  # the max step
        self.num_samples = num_samples
        self.sampled_actions = generate_unique_samples(self.size, num_samples, self.F)
    def get_initial_state(self):
        ## classic strassen's program
        state0 = np.zeros((self.size, self.size, self.size))
        state0[0,0:self.n,0:self.n] = np.eye(self.n)
        state0[1,self.n:2*self.n,0:self.n] = np.eye(self.n)
        state0[2,0:self.n,self.n:2*self.n] = np.eye(self.n)
        state0[3,self.n:2*self.n,self.n:2*self.n] = np.eye(self.n)
        return state0
    def get_next_state(self, state, action):
        ## s <-- s - u x v x w
        state -= np.einsum('i,j,k->ijk', action[0], action[1], action[-1])
        return state
    def get_valid_move(self):
        ## the policy to choose move, firstly at random because of not study
        return random.choice(self.sampled_actions)
    def get_valid_moves(self, n):
        ## sample moves with number n
        moves = [self.get_valid_move() for i in range(n)]
        return moves
    def check_win(self, state):
        ## check whether the state equalling 0 tensor
        return np.all(state == 0)
    def get_value_and_terminated(self, state, t):
        if self.check_win(state):
            return 1, True
        if t >= self.R_limit:
            return -1 - self.rank(state), True
        return -1, False
    def rank(self, state):
        ## a vague program to request the rank of 3-d tensor
        shape = state.shape
        assert len(shape) == 3, "program to only request the rank of 3-d tensor"
        ranks = [np.linalg.matrix_rank(state[:, :, i]) for i in range(shape[2])]
        return max(ranks)


if __name__ == "__main__":
    num_samples = 100
    robot = Decompositor(num_samples)
    state = robot.get_initial_state()
    print(f"state0:\n", state)
    t = 0
    # s1 = robot.get_next_state(s0, robot.get_valid_move())
    # print(s1, robot.check_win(s1), robot.get_value_and_terminated(s1))
    while t < robot.R_limit:
        action = robot.get_valid_move()
        state = robot.get_next_state(state, action)
        t += 1
        value, is_terminal = robot.get_value_and_terminated(state, t)
        if is_terminal:
            print(f"S{t},repay={value}:\n", state)
            break
        print(f"S{t},repay={value}:\n", state)