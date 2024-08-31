import numpy as np

original_array = np.load("/data/pulkit/train/EfficientGCNv1/data/npy_dataset/original/ntu-xsub120/eval_data.npy")
# Assuming your original array is named 'original_array'
# original_array.shape = (50919, 3, 300, 25, 2)

# Mapping of keypoints from 25 to 14
keypoint_mapping = {
    0: 4,
    1: 3,
    2: 9,
    3: 10,
    4: 24,
    5: 5,
    6: 6,
    7: 22,
    8: 17,
    9: 18,
    10: 20,
    11: 13,
    12: 14,
    13: 16
}

# Create a new array with the desired shape
new_array = np.zeros((original_array.shape[0], original_array.shape[1], original_array.shape[2], 14, original_array.shape[4]))

for i, l in keypoint_mapping.items():
    new_array[:, :, :, i] = original_array[:, :, :, l]

np.save('/data/pulkit/train/EfficientGCNv1/data/npy_dataset/original/ntu-xsub120/original_eval_data.npy', original_array)
np.save('/data/pulkit/train/EfficientGCNv1/data/npy_dataset/original/ntu-xsub120/eval_data.npy', new_array)


original_array = np.load("/data/pulkit/train/EfficientGCNv1/data/npy_dataset/original/ntu-xsub120/train_data.npy")
# Assuming your original array is named 'original_array'
# original_array.shape = (50919, 3, 300, 25, 2)

# Mapping of keypoints from 25 to 14
keypoint_mapping = {
    0: 4,
    1: 3,
    2: 9,
    3: 10,
    4: 24,
    5: 5,
    6: 6,
    7: 22,
    8: 17,
    9: 18,
    10: 20,
    11: 13,
    12: 14,
    13: 16
}

# Create a new array with the desired shape
new_array = np.zeros((original_array.shape[0], original_array.shape[1], original_array.shape[2], 14, original_array.shape[4]))

for i, l in keypoint_mapping.items():
    new_array[:, :, :, i] = original_array[:, :, :, l]

np.save('/data/pulkit/train/EfficientGCNv1/data/npy_dataset/original/ntu-xsub120/original_train_data.npy', original_array)
np.save('/data/pulkit/train/EfficientGCNv1/data/npy_dataset/original/ntu-xsub120/train_data.npy', new_array)
