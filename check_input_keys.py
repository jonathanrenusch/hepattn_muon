from hepattn.experiments.atlas_muon.data import AtlasMuonDataset
import yaml

config_path = "/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/configs/NGT/atlas_muon_event_NGT_plotting.yaml"
data_dir = "/scratch/ml_test_data_156000_hdf5"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

data_config = config.get('data', {})
inputs = {k: list(v) for k, v in data_config.get('inputs', {}).items()}
targets = {k: list(v) for k, v in data_config.get('targets', {}).items()}

dataset = AtlasMuonDataset(
    dirpath=data_dir,
    inputs=inputs,
    targets=targets,
    hit_eval_path=None
)

event = dataset[42]
inputs_batch, targets_batch = event

print("Input keys:", list(inputs_batch.keys()))
print("\nKeys containing 'x':", [k for k in inputs_batch.keys() if 'x' in k.lower()])
print("Keys containing 'y':", [k for k in inputs_batch.keys() if 'y' in k.lower()])
print("Keys containing 'z':", [k for k in inputs_batch.keys() if 'z' in k.lower()])
print("Keys containing 'truth':", [k for k in inputs_batch.keys() if 'truth' in k.lower()])
