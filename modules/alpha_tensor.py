import torch

from extras import (
    QuantileLoss,
    ValueRiskManagement,
)
from heads import PolicyHead, ValueHead
from torso import TorsoModel


class AlphaTensorModel(torch.nn.Module):
    def __init__(
        self,
        tensor_length: int,
        input_size: int,
        scalars_size: int,
        emb_dim: int,
        n_steps: int,
        n_logits: int,
        n_samples: int,
    ):
        # scalar_size = s
        # input_size = S
        # tensor_length = T
        # emb_dim = c
        super().__init__()
        self.tensor_length = tensor_length
        self.input_size = input_size
        self.emb_dim = emb_dim
        self.torso = TorsoModel(
            scalars_size, input_size, tensor_length, emb_dim
        )
        emb_size = 3 * input_size * input_size
        print("Build policy head")
        self.policy_head = PolicyHead(
            emb_size, emb_dim, n_steps, n_logits, n_samples
        )
        print("Build value head")
        self.value_head = ValueHead(
            2048
        )  # value dependent on num_head and proj_dim
        self.policy_loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        self.quantile_loss_fn = QuantileLoss()
        self.risk_value_management = ValueRiskManagement()

    @property
    def device(self):
        return next(self.parameters()).device

    def _train_forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        g_action: torch.Tensor,
        g_value: torch.Tensor,
    ):
        # shapes
        # x = (N, T, S, S, S)
        # s = (N, s)
        # g_action = (N, N_steps)
        # g_value = (N, )
        e = self.torso(x, s)
        o, z1 = self.policy_head(e, g_action)
        l_policy = self.policy_loss_fn(
            o.reshape(-1, o.shape[-1]), g_action.reshape(-1)
        )
        q = self.value_head(z1)
        l_value = self.quantile_loss_fn(q, g_value.float())
        return l_policy, l_value

    def _eval_forward(self, x: torch.Tensor, s: torch.Tensor):
        e = self.torso(x, s)
        a, p, z1 = self.policy_head(e)
        q = self.value_head(z1)
        q = self.risk_value_management(q)
        return a, p, q

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        g_action: torch.Tensor = None,
        g_value: torch.Tensor = None,
    ):
        if g_action is None:
            return self._eval_forward(x, s)
        else:
            assert g_value is not None
            return self._train_forward(x, s, g_action, g_value)

    @property
    def n_logits(self):
        return self.policy_head.n_logits

    @property
    def n_steps(self):
        return self.policy_head.n_steps

    @property
    def n_samples(self):
        return self.policy_head.n_samples



if __name__ == "__main__":
    # test this code
    # init the model
    matrix_size = 2
    tensor_length = 4
    input_size = matrix_size ** 2
    scalar_size = 1
    emb_dim = 256
    n_steps = 3
    n_actions = 3 ** (3 * input_size // n_steps)
    n_logits = n_actions
    n_samples = 33
    alphatensor = AlphaTensorModel(
        tensor_length=tensor_length,
        input_size=input_size,
        scalars_size=scalar_size,
        emb_dim=emb_dim,
        n_steps=n_steps,
        n_logits=n_logits,
        n_samples=n_samples 
    )
    # simulation
    N = 12
    x = torch.randint(-1,1,(N,tensor_length,input_size,input_size,input_size),dtype=torch.float32)
    s = torch.randint(0,10,(N,scalar_size),dtype=torch.float32)
    # # eval
    # a, p, q = alphatensor(x,s)
    # print(a,p,q)
    # print(a.shape,p.shape,q.shape)

    # train
    g_action = torch.randint(0,n_logits-1,size=(N,n_steps),dtype=torch.long)
    g_value = torch.randn(size=(N,1),dtype=torch.float32)
    # g_value = g_value.expand((-1,8))
    l_policy, l_value = alphatensor(x,s,g_action,g_value)
    print(l_policy,l_value)
    print(l_policy.shape,l_value.shape)
    

    
    
