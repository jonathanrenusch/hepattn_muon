from hepattn.experiments.atlas_muon.data import MuonTracking



def main():
    dataset = MuonTracking(
        dirpath="/home/iwsatlas1/jrenusch/master_thesis/tracking/data/train_setPU200",
        inputs={"input1": "value1", "input2": "value2"},
        targets={"target1": "value1", "target2": "value2"},
        num_events=1000,) 
    dataset.inspect_root_files()


if __name__ == "__main__":
    main()