import torch.utils
from modules.alpha_tensor import AlphaTensorModel
from training import Trainer
import torch
from pathlib import Path
import torch.nn as nn

def main():
    # init the model
    matrix_size = 2
    tensor_length = 3
    input_size = matrix_size ** 2
    scalar_size = 1
    emb_dim = 512
    n_steps = 3
    n_actions = 5 ** (3 * input_size // n_steps)
    n_logits = n_actions
    n_samples = 32
    alphatensor = AlphaTensorModel(
        tensor_length=tensor_length+1,
        input_size=input_size,
        scalars_size=scalar_size,
        emb_dim=emb_dim,
        n_steps=n_steps,
        n_logits=n_logits,
        n_samples=n_samples 
    )
    # init the trainer
    batch_size = 16
    optimizer = torch.optim.Adam(
        alphatensor.parameters(),
        lr=1e-4, weight_decay=1e-5
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        model=alphatensor,tensor_size=input_size,
        n_steps=n_steps, batch_size=batch_size,
        optimizer=optimizer, device=device,
        len_data=512, pct_synth=0.9,
        n_synth_data=20000, limit_rank=8,
        n_cob=1000, cob_prob=0.9983,
        data_augmentation=False,   # 关于数据增强,需要再看看
        loss_params=(1,1), random_seed=722,
        checkpoint_dir="work/AlphaTensor/record/Checkpoint",
        checkpoint_data_dir=Path("work/AlphaTensor/record/Data")
    )
    # training
    n_epochs = 1000
    n_games = 1
    mc_n_sim = 64
    N_bar = 10
    initial_lr = 1e-4
    lr_decay_factor = 0.1
    lr_decay_steps = 25
    trainer.train(
        n_epochs=n_epochs,
        n_games=n_games,
        mc_n_sim=mc_n_sim,
        N_bar=N_bar,
        initial_lr=initial_lr,
        lr_decay_factor=lr_decay_factor,
        lr_decay_steps=lr_decay_steps
    )




if __name__ == "__main__":
    # test this code
    # init the model
    main()
    
