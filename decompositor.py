## A program to simulate the process of matric decomposition
## Just considerring the Strassen first
import numpy as np

class Decompositor:
    def __init__(self):
        self.n = 2  # T_{m,n,p} with m=n=p=2
        self.size = self.n * self.n  # the size of T2
        self.F = (-1,0,1)  # the only coefficient to be used
        self.gamma = 1  # upper bound
        self.R_limit = self.n ** 3  # the max rank of the matric we need
        self.actions = []
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
        u = np.random.choice(self.F, self.size)
        v = np.random.choice(self.F, self.size)
        w = np.random.choice(self.F, self.size)
        self.actions.append((u,v,w))
        return (u,v,w)
    def check_win(self, state):
        ## check whether the state equalling 0 tensor
        return np.all(state == 0)
    def get_value_and_terminated(self, state):
        if self.check_win(state):
            return 1, True
        if len(self.actions) >= self.R_limit:
            return -self.rank(state), True
        return -1, False
    def rank(self, state):
        ## a vague program to request the rank of 3-d tensor
        shape = state.shape
        assert len(shape) == 3, "program to only request the rank of 3-d tensor"
        ranks = [np.linalg.matrix_rank(state[:, :, i]) for i in range(shape[2])]
        return max(ranks)



if __name__ == "__main__":
    robot = Decompositor()
    state = robot.get_initial_state()
    # s1 = robot.get_next_state(s0, robot.get_valid_move())
    # print(s1, robot.check_win(s1), robot.get_value_and_terminated(s1))
    count = 0
    while True:
        print(f"state{count}:\n", state,robot.get_value_and_terminated(state))
        count += 1
        action = robot.get_valid_move()
        state = robot.get_next_state(state, action)
        value, is_terminal = robot.get_value_and_terminated(state)
        if is_terminal:
            print("The terminated state:\n", state, robot.get_value_and_terminated(state))
            break
        
