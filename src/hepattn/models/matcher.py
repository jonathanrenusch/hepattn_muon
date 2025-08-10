import time

import scipy
import torch
from torch import nn
import numpy as np 
import lap1015


import lap

def solve_lap(cost):
    # cost must be float64 and C-contiguous for lap.lapjv
    cost = np.ascontiguousarray(cost, dtype=np.float64)
    _, col_idx, _ = lap.lapjv(cost)
    return col_idx

def solve_scipy(cost):
    _, col_idx = scipy.optimize.linear_sum_assignment(cost)
    return col_idx


def solve_1015_early(cost):
    return lap1015.lap_early(cost)


def solve_1015_late(cost):
    return lap1015.lap_late(cost)


SOLVERS = {
    "scipy": solve_scipy,
    # "1015_early": solve_1015_early,
    "1015_late": solve_1015_late,
}


class Matcher(nn.Module):
    def __init__(
        self,
        # default_solver: str = "scipy",
        default_solver: str = "1015_late",
        adaptive_solver: bool = True,
        adaptive_check_interval: int = 1000,
    ):
        super().__init__()
        """ Used to match predictions to targets based on a given cost matrix.

        Parameters
        ----------
        default_solver: str
            The default solving algorithm to use.
        adaptive_solver: bool
            If true, then after every adaptive_check_interval calls of the solver,
            each solver algorithm is timed and used to determine the fastest solver, which
            is then set as the current solver.
        adaptive_check_interval: bool
            Interval for checking which solver is the fastest.
        """
        if default_solver not in SOLVERS:
            raise ValueError(f"Unknown solver: {default_solver}. Available solvers: {list(SOLVERS.keys())}")
        self.solver = SOLVERS[default_solver]
        self.adaptive_solver = adaptive_solver
        self.adaptive_check_interval = adaptive_check_interval
        self.step = 0

    # def compute_matching(self, costs, object_valid_mask=None):
    #     # costs: numpy array [batch, pred, true]
    #     if object_valid_mask is None:
    #         object_valid_mask = np.ones((costs.shape[0], costs.shape[1]), dtype=bool)
    #     else:
    #         # Ensure torch tensors are moved to CPU before converting to numpy
    #         if isinstance(object_valid_mask, torch.Tensor):
    #             object_valid_mask = object_valid_mask.detach().cpu().numpy().astype(bool)
    #         else:
    #             object_valid_mask = np.asarray(object_valid_mask, dtype=bool)

    #     batch_obj_lengths = np.sum(object_valid_mask, axis=1).reshape(-1, 1)
    #     print("batch_obj_lengths:", batch_obj_lengths)
    #     idxs = []
    #     default_idx = np.arange(costs.shape[2])

    #     for k in range(len(costs)):
    #         num_valid = int(batch_obj_lengths[k][0])
    #         cost = costs[k][:, :num_valid].T
    #         if num_valid == 1:
    #             min_idx = np.argmin(cost, axis=1)
    #             pred_idx = np.full((cost.shape[1],), -1, dtype=np.int64)
    #             pred_idx[min_idx] = 0
    #         else:
    #             pred_idx = self.solver(cost)
    #             if self.solver == SOLVERS["scipy"]:
    #                 pred_idx = np.concatenate([pred_idx, default_idx[~np.isin(default_idx, pred_idx)]])
    #         print("This is prediction index:", pred_idx)
    #         idxs.append(pred_idx)

    #     pred_idxs = np.stack(idxs)
    #     pred_idxs = torch.from_numpy(pred_idxs)
    #     return pred_idxs
    
    # def compute_matching(self, costs, object_valid_mask=None):
    #     if object_valid_mask is None:
    #         object_valid_mask = torch.ones((costs.shape[0], costs.shape[1]), dtype=bool)

    #     object_valid_mask = object_valid_mask.detach().bool()
    #     batch_obj_lengths = torch.sum(object_valid_mask, dim=1).unsqueeze(-1)
    #     # print("batch_obj_lengths:", batch_obj_lengths)
    #     idxs = []
    #     default_idx = torch.arange(costs.shape[2])

    #     # Do the matching sequentially for each example in the batch
    #     for k in range(len(costs)):
    #         # remove invalid targets for efficiency
    #         cost = costs[k][:, : batch_obj_lengths[k]].T
    #         pred_idx = torch.as_tensor(self.solver(cost), device=cost.device)
    #         # print("num of valid:", num_valid)
    #         # print("pred_idx:", pred_idx)

    #         # scipy returns incomplete assignments, handle that here
    #         if self.solver == SOLVERS["scipy"]:
    #             pred_idx = torch.concatenate([pred_idx, default_idx[~torch.isin(default_idx, pred_idx)]])
    #         print("Concatenated pred idx:", pred_idx)
    #         # These indicies can be used to permute the predictions so they now match the truth objects
    #         idxs.append(pred_idx)

    #     pred_idxs = torch.stack(idxs)
    #     return pred_idxs
    def compute_matching(self, costs, object_valid_mask=None):
        if object_valid_mask is None:
            object_valid_mask = np.ones((costs.shape[0], costs.shape[1]), dtype=bool)
        else:
            # Ensure torch tensors are moved to CPU before converting to numpy
            print("We have got a mask!")
            if isinstance(object_valid_mask, torch.Tensor):
                object_valid_mask = object_valid_mask.detach().cpu().numpy().astype(bool)
            else:
                object_valid_mask = np.asarray(object_valid_mask, dtype=bool)

        batch_obj_lengths = np.sum(object_valid_mask, axis=1)
        print("batch_obj_lengths:", batch_obj_lengths)  
        idxs = []
        default_idx = np.arange(costs.shape[2])
        print("These are the costs:", costs)
        # Do the matching sequentially for each example in the batch
        for k in range(len(costs)):
            # Get the number of valid objects as a scalar
            num_valid = int(batch_obj_lengths[k])
            
            if num_valid == 1:
                # Special case: only one valid target
                # Find which prediction has the lowest cost for the single target
                cost_for_target = costs[k][:, 0]  # costs for the single valid target
                best_pred_idx = np.argmin(cost_for_target)
                
                # Create assignment: put the best prediction at index 0, others fill remaining slots
                pred_idx = np.arange(costs.shape[2])  # [0, 1, 2, 3, 4, 5]
                # Move the best prediction to position 0
                pred_idx[0] = best_pred_idx
                pred_idx[best_pred_idx] = 0
                
            else:
                # remove invalid targets for efficiency
                cost = costs[k][:, :num_valid].T
                cost = costs[k][:, :num_valid].T

                # Check cost matrix validity before passing to solver
                if cost.ndim != 2:
                    print(f"ERROR: Cost matrix is not 2D! Shape: {cost.shape}")
                    raise ValueError("Cost matrix must be 2D for assignment solver.")

                if np.isnan(cost).any():
                    print(f"ERROR: Cost matrix contains NaNs! Shape: {cost.shape}")
                    raise ValueError("Cost matrix contains NaNs.")

                if np.isinf(cost).any():
                    print(f"ERROR: Cost matrix contains infs! Shape: {cost.shape}")
                    raise ValueError("Cost matrix contains infs.")

                if cost.shape[0] == 0 or cost.shape[1] == 0:
                    print(f"ERROR: Cost matrix has zero dimension! Shape: {cost.shape}")
                    raise ValueError("Cost matrix must have nonzero dimensions.")

                if cost.shape[0] < cost.shape[1]:
                    print(f"WARNING: More targets than predictions! Shape: {cost.shape}")

                # Optionally, check for extreme values
                if np.abs(cost).max() > 1e8:
                    print(f"WARNING: Cost matrix has very large values! Max: {np.abs(cost).max()}")

                # ...existing code...
                pred_idx = self.solver(cost)
                print("This is pred_idx:", pred_idx)
                # scipy returns incomplete assignments, handle that here
                if self.solver == SOLVERS["scipy"]:
                    pred_idx = np.concatenate([pred_idx, default_idx[~np.isin(default_idx, pred_idx)]])
            print("Concatenated pred idx:", pred_idx)
            # These indices can be used to permute the predictions so they now match the truth objects
            idxs.append(pred_idx)

        pred_idxs = np.stack(idxs)
        # Convert to torch tensor at the end
        pred_idxs = torch.from_numpy(pred_idxs)
        return pred_idxs
    
    # def compute_matching(self, costs, object_valid_mask=None):
    #     if object_valid_mask is None:
    #         object_valid_mask = torch.ones((costs.shape[0], costs.shape[1]), dtype=bool)

    #     object_valid_mask = object_valid_mask.detach().bool()
    #     batch_obj_lengths = torch.sum(object_valid_mask, dim=1).unsqueeze(-1)
    #     idxs = []
    #     default_idx = torch.arange(costs.shape[2])

    #     # Do the matching sequentially for each example in the batch
    #     for k in range(len(costs)):
    #         num_valid = batch_obj_lengths[k].item()
    #         # remove invalid targets for efficiency
    #         cost = costs[k][:, :num_valid].T
    #         pred_idx = torch.as_tensor(self.solver(cost), device=cost.device)

    #         # scipy returns incomplete assignments, handle that here
    #         if self.solver == SOLVERS["scipy"]:
    #             pred_idx = torch.concatenate([pred_idx, default_idx[~torch.isin(default_idx, pred_idx)]])
            
    #         # These indices can be used to permute the predictions so they now match the truth objects
    #         idxs.append(pred_idx)

    #     pred_idxs = torch.stack(idxs)
    #     return pred_idxs

    @torch.no_grad()
    def forward(self, costs, object_valid_mask=None):
        # Cost matrix dimensions are batch, pred, true
        # Solvers need numpy arrays on the cpu
        costs = costs.detach().to(torch.float32).cpu().numpy()

        # If we are at a check interval, use the current cost batch to see which
        # solver is the fastest, and set that to be the new solver
        if self.adaptive_solver and self.step % self.adaptive_check_interval == 0:
            self.adapt_solver(costs)
        print(type(costs), "costs type")
        pred_idxs = self.compute_matching(costs, object_valid_mask)
        self.step += 1
        print(torch.sum(pred_idxs < 0), "negative indices in pred_idxs")
        print(torch.sum(pred_idxs >= 0), "positive indices in pred_idxs")
        assert torch.all(pred_idxs >= 0), "Matcher error!"
        return pred_idxs

    def adapt_solver(self, costs):
        solver_times = {}

        # For each solver, compute the time to match the entire batch
        for solver_name, solver in SOLVERS.items():
            # Switch to the solver we are testing
            self.solver = solver
            start_time = time.time()
            self.compute_matching(costs)
            solver_times[solver_name] = time.time() - start_time

        # Get the solver that was the fastest
        fastest_solver = min(solver_times, key=solver_times.get)

        # Set the new solver to be the solver with the fastest time for the cost batch
        self.solver = SOLVERS[fastest_solver]
