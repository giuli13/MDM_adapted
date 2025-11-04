import numpy as np

gen_length = 120 # frames

n_frames_dict = np.load('ata_eval/per_object_n_frames.npy', allow_pickle=True).item()
char_names = list(n_frames_dict.keys())
n_frames = np.array([c['total_frames'] for c in n_frames_dict.values()])
filter = np.logical_and(gen_length*5 < n_frames, n_frames < gen_length*10)
filtered_char = [c + '\n' for c, f in zip(char_names, filter) if f]
filtered_char[-1] = filtered_char[-1].strip()

out_path = 'ata_eval/benchmark_names.txt'
with open(out_path, 'w') as wp:
    wp.writelines(filtered_char)

print (f'created [{filter.sum()}] characters benchmark in [{out_path}]')