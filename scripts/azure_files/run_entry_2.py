import subprocess
import os
import sys
import time



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
    N = sys.argv[13]
    sigma = sys.argv[14]
    target_action = sys.argv[15]
    wandb_key = sys.argv[16]
    hugginface_key = sys.argv[17]
    som_origin = sys.argv[18]
    a11y_backend = sys.argv[19]

    # print all args
    print("Arguments:")
    arg_names = [
        "storage_path", "mounted_output_path", "exp_name", "num_workers", "worker_id", "agent", "json_name",
        "model_name", "run_mode", "epsilon", "alpha", "num_steps", "N", "sigma", "target_action",
        "wandb_key", "hugginface_key", "som_origin", "a11y_backend"
    ]
    for i, name in enumerate(arg_names, start=1):
        print(f"  {name}: {sys.argv[i]}")

    print("Starting entry script...")

    # print("Folder at ./:")
    # subprocess.run(['ls', '-la'], check=True)

    # create a folder with the specified path and experiment name
    result_dir = os.path.join(mounted_output_path, exp_name)
    os.makedirs(result_dir, exist_ok=True)
    print(f"Folder created: {result_dir}")
    # create a simple file with the current date and time in both the file name and content
    with open(os.path.join(result_dir, f"output_{exp_name}_worker_{str(worker_id)}.txt"), "w") as f:
        f.write(f"Output file created at {time.ctime()}")
    
    # a few debugging commands  
    # try:  
    #     subprocess.check_call(["ip", "link", "add", "dummy0", "type", "dummy"])  
    #     print("NET_ADMIN capability is enabled")  
    # except subprocess.CalledProcessError:  
    #     print("NET_ADMIN capability is not enabled")  
    
    # print("------------------\n\n")  
    
    # try:    
    #     subprocess.check_call(["iptables", "-A", "INPUT", "-p", "tcp", "--dport", "9999", "-j", "ACCEPT"])  
    #     print("NET_ADMIN capability is enabled")  
    #     subprocess.check_call(["iptables", "-D", "INPUT", "-p", "tcp", "--dport", "9999", "-j", "ACCEPT"])  
    # except subprocess.CalledProcessError:    
    #     print("NET_ADMIN capability is not enabled")   

    # launches the client script    
    # print("display the content of /client")  
    # os.system("ls -l /client")  
    os.system(f"cd /client && python {run_mode} --agent_name {agent} --worker_id {worker_id} --num_workers {num_workers} --result_dir {result_dir} --test_all_meta_path {json_name} --model {model_name} --som_origin {som_origin} --a11y_backend {a11y_backend} --epsilon {epsilon} --alpha {alpha} --num_steps {num_steps} --N {N} --sigma {sigma} --target_action {target_action} --wandb_key {wandb_key} --hugginface_key {hugginface_key}")

    print("Finished running entry script")



if __name__ == "__main__":
    main()