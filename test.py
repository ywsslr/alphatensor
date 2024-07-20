from modules.alpha_tensor import AlphaTensorModel
import torch
from modules.heads import PolicyHead, ValueHead
from actor.stage import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_init_state(tensor_length):
    size = 4
    n=2
    state0 = torch.zeros((size, size, size))
    state0[0,0:n,0:n] = torch.eye(n)
    state0[1,n:2*n,0:n] = torch.eye(n)
    state0[2,0:n,n:2*n] = torch.eye(n)
    state0[3,n:2*n,n:2*n] = torch.eye(n)
    history = tensor_length - 1
    return torch.cat((state0.reshape(1,1,size,size,size),torch.zeros((1,history, size,size,size))),dim=1)


def test_alpha_tensor():
    # init the model
    matrix_size = 2
    tensor_length = 5
    input_size = matrix_size ** 2
    scalar_size = 1
    emb_dim = 256
    n_steps = 3
    n_actions = 5 ** (3 * input_size // n_steps)
    n_logits = n_actions
    n_samples = 32
    alphatensor = AlphaTensorModel(
        tensor_length=tensor_length,
        input_size=input_size,
        scalars_size=scalar_size,
        emb_dim=emb_dim,
        n_steps=n_steps,
        n_logits=n_logits,
        n_samples=n_samples 
    )
    # # simulation
    # N = 12
    # x = torch.randint(-1,1,(N,tensor_length,input_size,input_size,input_size),dtype=torch.float32)  # states
    # s = torch.randint(0,10,(N,scalar_size),dtype=torch.float32)  # scalars
    # # eval
    # a, p, q = alphatensor(x,s)
    # print(a,p,q)
    # print(a.shape,p.shape,q.shape)

    # # train
    # g_action = torch.randint(0,n_logits-1,size=(N,n_steps),dtype=torch.long)
    # g_value = torch.randn(size=(N,1),dtype=torch.float32)
    # # g_value = g_value.expand((-1,8))
    # l_policy, l_value = alphatensor(x,s,g_action,g_value)
    # print(l_policy,l_value)
    return alphatensor

def test_heads():
    matrix_size = 2
    tensor_length = 4
    input_size = matrix_size ** 2
    scalar_size = 1
    emb_dim = 512
    emb_size = 3*input_size*input_size
    n_steps = 3
    n_actions = 3 ** (3 * input_size // n_steps)
    n_logits = n_actions
    n_samples = 33
    batch_size = 16
    policy_head = PolicyHead(emb_size, emb_dim, n_steps, n_logits, n_samples)
    value_head = ValueHead(2048)
    # simulation
    # # policy_head
    # # train
    # e = torch.randn((batch_size, 3*input_size*input_size, emb_dim))
    # g = torch.randint(0,n_logits-1,(batch_size, n_steps))
    # o,z = policy_head(e,g)
    # print(o,z)
    # print(o.shape,z.shape)

    # eval
    e = torch.randn((batch_size, 3*input_size*input_size, emb_dim))
    future_g,ps,z = policy_head(e)
    print(future_g,ps,z)
    print(future_g.shape,ps.shape,z.shape)

    # # value head
    # v = value_head(z)
    # print(v)
    # print(v.shape)

def test_mcts():
    torch.manual_seed(42)
    alphatensor = test_alpha_tensor().to(device)
    # state = torch.randint(-2,2,size=(1,5,4,4,4)).to(device)
    state = generate_init_state(5)

    # # actions
    # actions = actor_prediction(
    #     alphatensor, input_tensor=state,
    #     maximum_rank=8, mc_n_sim=200,
    #     N_bar=100, return_actions=True
    # )
    # print(actions)
    # print(len(actions))
    # print(actions[0])

    # # states, policies, rewards
    # states, policies, rewards = actor_prediction(
    #     alphatensor, input_tensor=state,
    #     maximum_rank=8, mc_n_sim=200,
    #     N_bar=100
    # )
    # print(states, policies, rewards)
    

    # MCTS
    rank = 0
    game_tree = {}
    state_dict = {}
    hash_states = []
    states = []
    states.append(state)
    hash_states.append(to_hash(extract_present_state(state)))
    state = monte_carlo_tree_search(
        alphatensor,
        state,
        50,
        rank,
        8,
        game_tree,
        state_dict,
    )
    print(state)
    print(state.shape)


if __name__ == "__main__":
    # test this code
    # test_alpha_tensor()
    # test_heads()
    test_mcts()

    # init_state = generate_init_state(5)
    # print(init_state)

