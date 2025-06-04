import json
import os
import shlex
import subprocess

def get_prompts(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    cmds = []
    img_path = "/home/cxx/HWs/AKS/Mixed Video Pipeline/Testing/outputs/Shooting/class_2/jpgs"
    
    for idx, d in enumerate(data[360:480]):
        path = d['video_path']
        id = os.path.basename(path).split('.')[0]
        vid = f"{id}_q{idx}"
        folder_path = os.path.join(img_path, vid)
        
        question = d['question']
        choices = d['choices']
        prompt = f"""You are a video checking expert, you will be given an image of video, you need to answer the following question: Q: {question} Your candidate answer is as follows: {choices}. You need to answer the question by giving one of the candidates, like Fighting or Shooting, Without any explanations in Force. If you find there aren't any answer can be found, you should give NONE as the answer."""
        
        k = 0
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") and filename.startswith("Test_"):
                img = os.path.join(folder_path, filename)
                k += 1
                if k == 4:
                    break
                
                command = [
                    "/home/cxx/HWs/AKS/llama.cpp/llama-minicpmv-cli",
                    "-m", "/home/cxx/HWs/AKS/llama.cpp/Minicpmv2_6_gguf/ggml-model-Q4_K_M.gguf",
                    "--mmproj", "/home/cxx/HWs/AKS/llama.cpp/Minicpmv2_6_gguf/mmproj-model-f16.gguf",
                    "-c", "4096",
                    "--temp", "0.7",
                    "--top-p", "0.8",
                    "--top-k", "100",
                    "--repeat-penalty", "1.05",
                    "--image", img,
                    "-p", prompt
                ]
                cmds.append(command)
    
    with open('/home/cxx/HWs/AKS/Mixed Video Pipeline/Testing/Shooting_class_2_cmd.txt', 'w') as f:
        for cmd in cmds:
            f.write(' '.join(shlex.quote(arg) for arg in cmd) + '\n')
    
    print(f"✅ Prompts saved to '/home/cxx/HWs/AKS/Mixed Video Pipeline/Testing/Shooting_class_2_cmd.txt'")
    return cmds

json_file = "/home/cxx/HWs/AKS/Mixed Video Pipeline/outputs/mixed_video_qa_randomized_4x_per_video.json"
cmds = get_prompts(json_file)

import subprocess
import shlex

# Read commands from file
with open('/home/cxx/HWs/AKS/Mixed Video Pipeline/Testing/Shooting_class_2_cmd.txt', 'r') as f:
    cmds = [line.strip() for line in f.readlines()]  # Strip newline characters
res_path = "/home/cxx/HWs/AKS/Mixed Video Pipeline/Testing/Shooting_class_2_res.txt"
res = []
group_size = 4
total_cmds = len(cmds)
# Iterate through each group
for i in range(2, total_cmds, group_size):
    # Extract the current group of commands
    group = cmds[i:i + group_size]
    
    # Process the first command in the group
    first_cmd = group[0]
    print(f"Running first command in group {i // group_size + 1}: {first_cmd}")
    # id = i // group_size + 1 # id of the current group 1~N
    try:
        # Split the command into a list of arguments
        cmd_list = shlex.split(first_cmd)
        # Run the command
        result = subprocess.run(cmd_list, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"result: {result.stdout.decode().strip()}")
        res.append(result.stdout.decode().strip())
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr.decode().strip()}")
with open(res_path, 'w') as f:
    f.write('\n'.join(res))
print(f"✅ Results saved to {res_path}")