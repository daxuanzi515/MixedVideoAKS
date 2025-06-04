# from mixed_strategy import VideoMixer
# import os

# if __name__ == "__main__":
#     Number = 30
#     class_type = 2
#     total_duration = 360
#     B_folder = "/home/cxx/HWs/AKS/datasets/XD_violence/ours/Benign/"

#     B_list = sorted([
#         os.path.join(B_folder, f)
#         for f in os.listdir(B_folder)
#         if f.endswith(".mp4")
#     ])[:Number]


#     M_list = [f"/home/cxx/HWs/AKS/datasets/XD_violence/ours/Fighting/Test_{i}_Fighting.mp4" for i in range(Number)]
#     O_list = [f"/home/cxx/HWs/AKS/Mixed Video Pipeline/outputs/class_2/Test_{i}_Mixed.mp4" for i in range(Number)]

#     for i in range(30):
#         print(f"[🚀] Processing sample Class {class_type}: {i}...")
#         print(f"[🤖] Benign video: {B_list[i]}, Malignant video: {M_list[i]}")
#         mixer = VideoMixer(
#             benign_video=B_list[i],
#             malignant_video=M_list[i],
#             output_path=O_list[i]
#         )
#         mixer.generate_mixed_video(class_type=class_type, total_duration=total_duration)


# import os
# import re

# def rename_videos(directory):
#     """
#     Rename video files in the specified directory from the format
#     'Test_i_Mixed_class1_360s.mp4' to 'Test_i_Mixed.mp4'.

#     Args:
#         directory (str): The directory containing the video files to rename.
#     """
#     # Regular expression to match the filename pattern
#     pattern = re.compile(r'(Test_\d+_Mixed)_class2_360s\.mp4')
#     i = 0
#     # List all files in the directory
#     for filename in os.listdir(directory):
#         i+=1
#         match = pattern.match(filename)
#         if match:
#             # Extract the new filename without the '_class1_360s' part
#             new_filename = f"{match.group(1)}.mp4"
#             # Construct the full paths for the old and new filenames
#             old_file_path = os.path.join(directory, filename)
#             new_file_path = os.path.join(directory, new_filename)
#             # Rename the file
#             os.rename(old_file_path, new_file_path)
#             print(f"Renamed '{filename}' to '{new_filename}'")
#         # if i==1:
#         #     break
# # Example usage
# if __name__ == "__main__":
#     video_directory = "/home/cxx/HWs/AKS/Mixed Video Pipeline/outputs/Shooting/class_2"
#     rename_videos(video_directory)




# import os
# import json

# def read_all_frame_jsons(folder_path, output_path=None):
#     merged_list = []

#     # 遍历所有 json 文件
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".json") and filename.startswith("Test_") and "_Mixed_q" in filename:
#             file_path = os.path.join(folder_path, filename)
#             try:
#                 with open(file_path, 'r') as f:
#                     data = json.load(f)
#                     if isinstance(data, list):
#                         merged_list.append(data)
#                     else:
#                         print(f"⚠️ Skip non-list file: {filename}")
#             except Exception as e:
#                 print(f"❌ Failed to read {filename}: {e}")
#     with open(output_path, 'w') as f:
#         json.dump(merged_list, f)
#     print(f"✅ Merged {len(merged_list)} files to {output_path}")

#     return merged_list

# frames_dir = "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Shooting/class_2/blip/scores"
# output_path = "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Shooting/class_2/blip/scores.json"
# all_frames = read_all_frame_jsons(frames_dir, output_path)
# print(f"Total frames collected: {len(all_frames)}")


# frames_dir = "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Shooting/class_2/blip/frames"
# output_path = "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Shooting/class_2/blip/frames.json"
# all_frames = read_all_frame_jsons(frames_dir, output_path)
# print(f"Total frames collected: {len(all_frames)}")
# print(all_frames[:10])



# import json
# ori = "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Fighting/class_1/blip/selected_frames/selected_frames.json"
# opt = "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Fighting/class_1/blip/optimized_frames/optimized_frames.json"

# with open(ori, 'r') as f:
#     data_1 = json.load(f)

# with open(opt, 'r') as f:
#     data_2 = json.load(f)

# print(f"Number of samples in ori: {len(data_1)}")
# print(f"Number of samples in opt: {len(data_2)}")

# if len(data_1) != len(data_2):
#     print("❗Mismatch in number of samples.")
# else:
#     print("✅ Same number of samples.")


# diff_stats = []

# for i, (a, b) in enumerate(zip(data_1, data_2)):
#     set_a, set_b = set(a), set(b)
#     added = sorted(list(set_b - set_a))
#     removed = sorted(list(set_a - set_b))

#     if added or removed:
#         diff_stats.append({
#             "index": i,
#             "added": added,
#             "removed": removed
#         })

# print(f"\nTotal different entries: {len(diff_stats)}")


# for d in diff_stats[:5]:  # 打印前 5 个不同的样本
#     print(f"\nSample #{d['index']}")
#     print(f"  ➕ Added in opt: {d['added']}")
#     print(f"  ➖ Removed from ori: {d['removed']}")

    
# def jaccard(a, b):
#     a, b = set(a), set(b)
#     if not a and not b:
#         return 1.0
#     return len(a & b) / len(a | b)

# scores = [jaccard(a, b) for a, b in zip(data_1, data_2)]
# print(f"\nAverage frame overlap (Jaccard): {sum(scores)/len(scores):.4f}")
import json
import os
import shlex
import subprocess

def get_prompts(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    cmds = []
    img_path = "/home/cxx/HWs/AKS/Mixed Video Pipeline/Testing/outputs/Fighting/class_2/jpgs"
    
    for idx, d in enumerate(data[120:240]):
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
    
    with open('/home/cxx/HWs/AKS/Mixed Video Pipeline/Testing/Fighting_class_2_cmd.txt', 'w') as f:
        for cmd in cmds:
            f.write(' '.join(shlex.quote(arg) for arg in cmd) + '\n')
    
    print(f"✅ Prompts saved to '/home/cxx/HWs/AKS/Mixed Video Pipeline/Testing/Fighting_class_2_cmd.txt'")
    return cmds

json_file = "/home/cxx/HWs/AKS/Mixed Video Pipeline/outputs/mixed_video_qa_randomized_4x_per_video.json"
cmds = get_prompts(json_file)

import subprocess
import shlex

# Read commands from file
with open('/home/cxx/HWs/AKS/Mixed Video Pipeline/Testing/Fighting_class_2_cmd.txt', 'r') as f:
    cmds = [line.strip() for line in f.readlines()]  # Strip newline characters
res_path = "/home/cxx/HWs/AKS/Mixed Video Pipeline/Testing/Fighting_class_2_res.txt"
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