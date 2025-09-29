def pad_and_concat(items: list[Tensor], target_size: tuple[int], pad_value) -> Tensor:
    """Takes a list of tensors, pads them to a given size, and then concatenates them along a new dimension at zero."""
    return torch.cat([pad_to_size(item, (1, *target_size), pad_value) for item in items], dim=0)

class AtlasMuonCollator:
    def __init__(self, dataset_inputs, dataset_targets, max_num_obj):
        self.dataset_inputs = dataset_inputs
        self.dataset_targets = dataset_targets
        self.max_num_obj = max_num_obj

    def __call__(self, batch):
        inputs, targets = zip(*batch, strict=False)
        # print(targets[0].keys())
        # print(type(inputs))
        # print(type(targets))

        hit_max_sizes = {}
        # print(self.dataset_inputs)
        for input_name in self.dataset_inputs:
            hit_max_sizes[input_name] = max(event[f"{input_name}_valid"].shape[-1] for event in inputs)
        # print(hit_max_sizes)
        batched_inputs = {}
        batched_targets = {}
        for input_name, fields in self.dataset_inputs.items():
            k = f"{input_name}_valid"
            batched_inputs[k] = pad_and_concat([i[k] for i in inputs], (hit_max_sizes[input_name],), False)

            # Some tasks might require to know hit padding info for loss masking
            batched_targets[k] = batched_inputs[k]

            for field in fields:
                k = f"{input_name}_{field}"
                batched_inputs[k] = pad_and_concat([i[k] for i in inputs], (hit_max_sizes[input_name],), 0.0)
        # if "particle_hit_valid" in targets[0].keys():
        #     size = (self.max_num_obj, hit_max_sizes["hit"])
        #     batched_targets["particle_hit_valid"] = pad_and_concat([t["particle_hit_valid"] for t in targets], size, False)
        
        for target_name, fields in self.dataset_targets.items():
            # print("This is target:", target_name)
            # print("Fields:", fields)
            if target_name == "particle":
                size = (self.max_num_obj,)
            
            # elif target_name == "hit":
            #     size = (hit_max_sizes[target_name],)
                # print(size)
            else:
                hit = target_name.split("_")[1]
                size = (self.max_num_obj, hit_max_sizes[hit])
            k = f"{target_name}_valid"
            batched_targets[k] = pad_and_concat([t[k] for t in targets], size, False)

            for field in fields:
                k = f"{target_name}_{field}"
                batched_targets[k] = pad_and_concat([t[k] for t in targets], size, torch.nan)
            # print(batched_targets.keys())
        # Batch the metadata
        batched_targets["sample_id"] = torch.cat([t["sample_id"] for t in targets], dim=-1)
        # for key in batched_inputs.keys():
            # print(f"Input {key} shape: {batched_inputs[key].shape}")
            # print(f"Input {key}: {batched_inputs[key]}")
        # for key in batched_targets.keys():
        #     print(f"Target {key}: {batched_targets[key]}")
        return batched_inputs, batched_targets

