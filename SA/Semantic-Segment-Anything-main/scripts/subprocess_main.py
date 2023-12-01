import subprocess
import os
import torch

data_folder = './data'
subfolders = [f.name for f in os.scandir(data_folder) if f.is_dir()]

for subfolder in subfolders:
    subfolder_path = os.path.join(data_folder, subfolder)
    seq_folders = [f.name for f in os.scandir(subfolder_path) if f.is_dir()]

    # Iterate over each sequence folder
    for seq_folder in seq_folders:
        seq_folder_path = os.path.join(subfolder_path, seq_folder)

        torch.cuda.empty_cache()
        # Construct the command to run inference.py
        command = f"python scripts/main_ssa_engine.py --data_dir={seq_folder_path} --out_dir={seq_folder_path} --world_size=1 --save_img --sam --ckpt_path=models/sam_vit_h.pth --light_mode "

        # Run the command using subprocess
        subprocess.run(command, shell=True)
