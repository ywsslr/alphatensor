# test the utils.py
import torch
from utils import _single_action_to_triplet, map_action_to_triplet

# _single_action_to_triplet
def test_single_action_to_triplet():
    action_val = 10
    basis = 5
    out_dim = 4
    bias = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    triplet = _single_action_to_triplet(
        action_val, basis, out_dim, bias, device
    )
    print(triplet)
    return triplet
def test_map_action_to_triplet():
    action_tensor = torch.randint(0,5**4-1,(12,3))
    triplets = map_action_to_triplet(action_tensor, 5,4)
    print(triplets)
    print(triplets.shape)
    return triplets

if __name__ == "__main__":
    # test_single_action_to_triplet()
    test_map_action_to_triplet()

