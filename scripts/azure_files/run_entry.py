import subprocess
import os
import sys
import time





import base64, io, requests
from typing import Union, List, Optional
from PIL import Image
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration
import os
import inspect
import re

import tiktoken
import time
import json
import re
import argparse
import datetime
import json
import logging
import os
import random
import sys
import shutil
import traceback
# import wandb

from tqdm import tqdm


import requests
import time

from threading import Event
import signal




def main():
    storage_path = sys.argv[1]
    mounted_output_path = sys.argv[2]
    exp_name = sys.argv[3]
    num_workers = sys.argv[4]
    worker_id = sys.argv[5]
    agent = sys.argv[6]
    json_name = sys.argv[7]
    model_name = sys.argv[8]
    run_mode = sys.argv[9]
    epsilon = sys.argv[10]
    alpha = sys.argv[11]
    num_steps = sys.argv[12]
    target_action = sys.argv[13]
    som_origin = sys.argv[14]
    a11y_backend = sys.argv[15]

    # print all args
    print("All args:")
    for arg in sys.argv:
        print(arg)

    print("Starting entry script...")

    print("Folder at ./:")
    subprocess.run(['ls', '-la'], check=True)

    # create a folder with the specified path and experiment name
    result_dir = os.path.join(mounted_output_path, exp_name)
    os.makedirs(result_dir, exist_ok=True)
    print(f"Folder created: {result_dir}")
    # create a simple file with the current date and time in both the file name and content
    with open(os.path.join(result_dir, f"output_{exp_name}_worker_{str(worker_id)}.txt"), "w") as f:
        f.write(f"Output file created at {time.ctime()}")
    
    # a few debugging commands  
    try:  
        subprocess.check_call(["ip", "link", "add", "dummy0", "type", "dummy"])  
        print("NET_ADMIN capability is enabled")  
    except subprocess.CalledProcessError:  
        print("NET_ADMIN capability is not enabled")  
    
    print("------------------\n\n")  
    
    try:    
        subprocess.check_call(["iptables", "-A", "INPUT", "-p", "tcp", "--dport", "9999", "-j", "ACCEPT"])  
        print("NET_ADMIN capability is enabled")  
        subprocess.check_call(["iptables", "-D", "INPUT", "-p", "tcp", "--dport", "9999", "-j", "ACCEPT"])  
    except subprocess.CalledProcessError:    
        print("NET_ADMIN capability is not enabled")   

    print("Display the contents of the storage_path folder")  
    os.system(f"ls -l {storage_path}")  
    
    print("Create /tmp/storage_tmp directory")
    os.makedirs("/tmp/storage_tmp", exist_ok=True)  
    
    print("create alias so that /storage points to /tmp/storage_tmp")  
    os.symlink("/tmp/storage_tmp", "/storage", target_is_directory=True)  
    
    print("start copying data from the mounted directory to /tmp/storage_tmp")  
    start_time = time.time()  
    os.system(f"cp -r {storage_path}/* /tmp/storage_tmp")  
    end_time = time.time()  
    total_time = end_time - start_time  
    print("Total time taken to copy: {} seconds".format(total_time))  
    
    print("display the content of /storage")  
    os.system("ls -l /storage")  

    # starts the VM and waits for it to fully load before proceeding
    os.system("/entry_setup.sh") # since it's in root we can just do /script.sh and don't need cd /

    # launches the client script
    os.system(f"cd /client && python {run_mode} --agent_name {agent} --worker_id {worker_id} --num_workers {num_workers} --result_dir {result_dir} --test_all_meta_path {json_name} --model {model_name} --som_origin {som_origin} --a11y_backend {a11y_backend} --epsilon {epsilon} --alpha {alpha} --num_steps {num_steps} --target_action {target_action}")

    print("Finished running entry script")



if __name__ == "__main__":
    main()