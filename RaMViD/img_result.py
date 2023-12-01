import numpy as np
from PIL import Image
import os
import sys

def unzip_images(folder_path, npz_files):
    for i,npz_path in enumerate(npz_files):
        samples = np.load(npz_path)['arr_0']
        for j,sample in enumerate(samples):
            print(sample.shape)
            for k,ind_sample in enumerate(sample):
                ind_sample = ind_sample.reshape(64,64,3)
                image = Image.fromarray(ind_sample)
                image.save(f"{folder_path}/sample_{j}_{k}.png")

def list_npz_files(folder_path):
    npz_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.npz'):
                file_path = os.path.join(root, file)
                npz_files.append(file_path)

    # Print the list of .npz file paths
    for file_path in npz_files:
        print(file_path)
    return npz_files

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py folder_path")
    else:
        folder_path = sys.argv[1]
        npz_files = list_npz_files(folder_path)
        unzip_images(folder_path,npz_files)
