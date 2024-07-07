## the program to play the alphatensor utilizing the monte carlo tree search(to be finished)
import numpy as np
import math
from decompositor import Decompositor


class Node:
    def __init__(self, game:Decompositor, args, state, t, parent=None, action_taken=None):
        """
        the basic unit of MCTS search, and the parameter t is necessary!!
        """
        self.game = game
        self.args = args
        self.state = state  # current tensor S_t
        self.parent = parent  # parent node
        self.action_taken = action_taken  # record the last action chosen to be the current S_t
        self.t = t  # current decompositing number, equalling to the parents number.
        
        self.children = []  # children node
        if t < self.game.R_limit:
            self.moves = game.sampled_actions  # all the sampled actions can be used 
        self.expandable_moves = list(range(self.game.num_samples))  # remark the state of choose of all actions

        self.visit_count = 0
        self.value_sum = 0
        
    def is_fully_expanded(self):
        """
        judge if a node has took all the actions 
        """
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0
    
    def select(self):
        """
        select a best child of the node with the highest score
        """
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        """
        !!! The most important, should be considered
        get the pucb score, with the C balancing the exploration and exploitng
        """
        q_value = child.value_sum / (child.visit_count + 1) / 2 
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count)) / (child.visit_count + 1)
    
    def expand(self):
        """
        generate new chiled nodes
        """
        for i in range(self.args['num_expands']):
            action_taken = self.expandable_moves.pop(0)
            action = self.moves[action_taken]
            
            child_state = self.state.copy()
            child_state = self.game.get_next_state(child_state, action)
            
            child = Node(self.game, self.args, child_state,self.t + 1, self, action_taken)
            self.children.append(child)
        return child
    
    def simulate(self):
        """
        simulate a total game rapidly by random policy
        """
        value, is_terminal = self.game.get_value_and_terminated(self.state, self.t)
        
        if is_terminal:
            return value
        
        rollout_state = self.state.copy()
        t_ = self.t
        while True:
            action = self.game.get_valid_move()
            rollout_state = self.game.get_next_state(rollout_state, action)
            t_ += 1
            value, is_terminal = self.game.get_value_and_terminated(rollout_state, t_)
            if is_terminal:
                return value    
            
            
    def backpropagate(self, value):
        """
        update thecounts and value
        """
        self.value_sum += value
        self.visit_count += 1
        
        if self.parent is not None:
            self.parent.backpropagate(value)  


class MCTS:
    def __init__(self, game:Decompositor, args):
        self.game = game
        self.args = args
        
    def search(self, state, t):
        """
        simulate the MCTS search with the start of current state to choose the best next action
        Procedure:
            1.select
            2.expand
            3.simulation
            4.backpropagate
        Notice: When we choose the next node, we should update its t

        Args:
            state (np.array): current tensor to be de composite
            t (int): record how many rank-1 tensors taken, with the beginning 0

        Returns:
            the probabilities of next actions
        """
        root = Node(self.game, self.args, state,t)
        
        for search in range(self.args['num_searches']):
            node:Node = root
            
            
            while node.is_fully_expanded():
                node = node.select() # type: ignore
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.t)
            
            if not is_terminal:
                node = node.expand()
                value = node.simulate()
                
            node.backpropagate(value)    
            
            
        action_probs = np.zeros(self.game.num_samples)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
        
if __name__ == "__main":
    num_samples = 100    
    robot = Decompositor(num_samples)
    args = {
        'C': 1.41,
        'num_searches': 10000,
        'num_expands': 5
    }
    mcts = MCTS(robot, args)
    state = robot.get_initial_state()
    t = 0
    while t < robot.R_limit:
        mcts_probs = mcts.search(state,t)
        action_taken = np.argmax(mcts_probs)
        action = robot.sampled_actions[action_taken]

        state = robot.get_next_state(state, action)
        t += 1
        value, is_terminal = robot.get_value_and_terminated(state, t)
        if is_terminal:
            print(f"S{t},repay={value}:\n", state)
            break
        print(f"S{t},repay={value}:\n", state)
        

