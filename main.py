import time
from numpy import dtype
import custom_transforms as ct
import tonic

import matplotlib.pyplot as plt
import numpy as np

# np.random.seed(255)

# nmnist_dataset_train = tonic.datasets.NMNIST(save_to='../../data', train=True)
# nmnist_dataset = tonic.datasets.NMNIST(save_to='../../data', train=False)
# ncaltech101_dataset = tonic.datasets.NCALTECH101(save_to='../../data')

# sensor_size = tonic.datasets.NCALTECH101.sensor_size
# events_foreground, label_frg = nmnist_dataset[9000]  # past: 1000
# events_background, label_bkg = ncaltech101_dataset[10]  # past: 10

# print("Merging NMNIST digit " + str(label_frg) +
#       " with N-CALTECH101 " + label_bkg)


dt = np.dtype([('x', 'i8'), ('y', 'i8'), ('t', 'i8'), ('p', 'i8')])
empty_background = np.array([(0, 0, 100, 0), (33, 33, 100, 0)], dtype=dt)

# SI = ct.Superimposed(background_events=events_background,
#                      foreground_events=events_foreground)
# dataset = SI.merge_datasets()

train_dataset_path = "../instance-segmentation/data/sorted_NMNIST/train_dataset.npy"
test_dataset_path = "../instance-segmentation/data/sorted_NMNIST/test_dataset.npy"
train_dataset, test_dataset = ct.get_sorted_datasets(train_dataset_path, test_dataset_path)
ncaltech101_dataset = tonic.datasets.NCALTECH101(save_to='../instance-segmentation/data')


# start_time = time.time()
# SI = ct.Superimposed(background_events=ncaltech101_dataset[5000][0],
#                      foreground_events=train_dataset[40000][0])
# dataset = SI.merge_datasets()
# print("--- Took %s seconds ---" % (time.time() - start_time))

ct.combine_datasets_parallel(test_dataset[:10], ncaltech101_dataset, "../instance-segmentation/data/alex_data/train_dataset_centereddropevents/part_")

# frames = ct.to_frames((34,34,2), dataset, time_bins=50)
# image = ct.to_image(SI.sensor_size, dataset[:50])

# ct.save_as_gif(frames, file_name="less_noisy_bkg.gif")


# print(type(dataset2))
# np.save("index_1000", dataset)

# print("frames:", frames.shape)
# print("image:", image.shape)
# ct.plot_frames(frames=frames)
