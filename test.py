from modules.alpha_tensor import AlphaTensorModel
import torch
from modules.heads import PolicyHead, ValueHead
from actor.stage import monte_carlo_tree_search, actor_prediction

device = "cuda" if torch.cuda.is_available() else "cpu"

def test_alpha_tensor():
    # init the model
    matrix_size = 2
    tensor_length = 3 
    input_size = matrix_size ** 2
    scalar_size = 1
    emb_dim = 256
    n_steps = 3
    n_actions = 3 ** (3 * input_size // n_steps)
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
    alphatensor = test_alpha_tensor().to(device)
    state = torch.randint(-2,2,size=(1,3,4,4,4)).to(device)
    # # actions
    # actions = actor_prediction(
    #     alphatensor, input_tensor=state,
    #     maximum_rank=10, mc_n_sim=200,
    #     N_bar=100, return_actions=True
    # )
    # states, policies, rewards
    states, policies, rewards = actor_prediction(
        alphatensor, input_tensor=state,
        maximum_rank=8, mc_n_sim=200,
        N_bar=100
    )
    print(states, policies, rewards)


if __name__ == "__main__":
    # test this code
    # test_alpha_tensor()
    # test_heads()
    test_mcts()

