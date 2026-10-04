import yaml
import dotmap
import lightning as L
import torch
from models.recognition import LNRecognition
from data.spiking_digits import SpikingDigits
from data.dataset import SpikingDS

dict_convert = { 0: 0,
                10: 1,
                20: 2,
                30: 3,
                40: 4,
                50: 5,
                60: 6,
                70: 7,
                80: 8,
                90: 9,
                100: 10,
                -100: 11,
                -90: 12,
                -80: 13,
                -70: 14,
                -60: 15,
                -50: 16,
                -40: 17,
                -30: 18,
                -20: 19,
                -10: 20,
}

cfg = yaml.load(open('configs/digits.yaml', 'r'), Loader=yaml.FullLoader)
cfg = dotmap.DotMap(cfg)

input = torch.load('datasets/hdspikes/processed/test/0.pt', weights_only=False)
print(input.pos)

dm = SpikingDS(files=['datasets/hdspikes/processed/test/0.pt'], config=cfg)

data = dm[0]

lm = LNRecognition.load_from_checkpoint('checkpoints/best_model_calibrated-v2.ckpt')


model = lm.model
model.eval()

out, _ = model(data.to('cuda'))
print(out)

model.quantize()

data = dm[0]

with open('model.txt', 'w') as f:
    for i, (x, pos) in enumerate(zip(data.x, data.pos)):
        t = pos[0]
        idx = pos[1]
        ts = (t*1000000+1e-3).round().item()
        ts = int(ts)
        idxs = int((idx*700).round().item())

        avg_t = x[0]
        avg_idx = x[1]

        q_avg_t = model.conv1.observer_input.quantize_tensor(avg_t)
        q_avg_idx = model.conv1.observer_input.quantize_tensor(avg_idx)

        # f.write(f'EVENT = {ts} {idxs} Avg_T={int(q_avg_t.item())} ({avg_t*1000000}) Avg_Idx={int(q_avg_idx.item())} ({avg_idx*700})\n')

        f.write(f'EVENT = {ts} {idxs}\n')
        
        mask = data.edge_index[1, :] == i
        neighbour_pos = data.pos[data.edge_index[0, mask].T]

        to_list = []

        if len(neighbour_pos) > 0:
            # Sort by idx from smallest to largest
            neighbour_pos = neighbour_pos[neighbour_pos[:, 1].argsort()]
            for n_pos in neighbour_pos:
                n_t = int((n_pos[0]*1000000).round().item())
                n_idx = int((n_pos[1]*700).round().item())

                diff_t = (n_pos[0] - t)*(-50)
                diff_idx = (n_pos[1] - idx + 1/7) * (7/2)
                diff_idx_converted = dict_convert[n_idx - idxs]

                to_list.append([diff_idx_converted, n_t - ts])

            
            # sort to list by diff_idx_converted
            to_list = sorted(to_list, key=lambda x: x[0])

            for diff_idx_converted, txd in to_list:
                f.write(f"Edge_{diff_idx_converted} = {txd}\n")
                # f.write(f'Diff T={n_t - ts} Diff Idx={n_idx - idxs} Quantized Diff T={int(model.conv1.observer_input.quantize_tensor(torch.tensor(diff_t)).item())} Quantized Diff Idx={int(model.conv1.observer_input.quantize_tensor(torch.tensor(diff_idx)).item())}\n')
