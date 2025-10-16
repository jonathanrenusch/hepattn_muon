import h5py

f = h5py.File('/scratch/epoch=139-val_loss=2.74982_ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit600_eval.h5', 'r')
print('First 5 keys in file:', list(f.keys())[:5])
if '42' in f:
    print('Keys in event 42:', list(f['42'].keys()))
    if 'preds' in f['42']:
        print('Preds keys:', list(f['42']['preds'].keys()))
        if 'final' in f['42']['preds']:
            print('Preds/final keys:', list(f['42']['preds']['final'].keys()))
    if 'outputs' in f['42']:
        print('Outputs keys:', list(f['42']['outputs'].keys()))
        if 'final' in f['42']['outputs']:
            print('Outputs/final keys:', list(f['42']['outputs']['final'].keys()))
else:
    print('Event 42 not found')
f.close()
