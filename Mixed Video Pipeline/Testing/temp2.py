import argparse
import os
import json
import shlex
import subprocess
from PIL import Image
from collections import Counter
# from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams

# # === 初始化模型一次 ===
# MODEL_PATH = "/home/cxx/HWs/AKS/checkpoints/MiniCPM-V-2_6"
# llm = LLM(model=MODEL_PATH, gpu_memory_utilization=1, trust_remote_code=True, max_model_len=2048)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# stop_tokens = ['<|im_end|>', '<|endoftext|>']
# stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
# sampling_params = SamplingParams(stop_token_ids=stop_token_ids, temperature=0, max_tokens=1024, best_of=1)

# === 构造 prompt ===
def build_prompt(question, choices):
    choice_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    return (
        "(<image>./</image>)\n"
        "You are analyzing a scene from a video. Carefully observe the image and select the most appropriate answer from the choices below.\n\n"
        f"Question:\n{question}\n\nChoices:\n{choice_str}\n\n"
        "Answer (choose only one option from above, e.g., \"A. Shooting\"):"
    )

# ffmpeg -i "/home/cxx/HWs/AKS/Mixed Video Pipeline/outputs/Fighting/class_1/Test_0_Mixed.mp4" -vf "select='eq(n,60)'" -vframes 1 "/home/cxx/HWs/AKS/Mixed Video Pipeline/Testing/outputs/Fighting/class_1/jpgs/Test_0_Mixed_60.jpg" -y

def extract_frame(video_path, frame_index, output_path):
    """
    使用 ffmpeg 从视频中截取指定帧并保存为图片。

    :param video_path: 视频文件路径
    :param frame_index: 需要截取的帧索引
    :param output_path: 保存截取图片的路径
    """
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return

    # 确保输出路径的父目录存在
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # 构造 ffmpeg 命令
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"select='eq(n,{frame_index})'",
        "-vframes", "1",
        output_path,
        "-y"  # 覆盖输出文件（如果已存在）
    ]
    print(f"Running command: {' '.join(command)}")
    # 调用 ffmpeg 命令
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"成功保存帧到: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"调用 ffmpeg 时出错: {e.stderr.decode().strip()}")
        print(f"调用 ffmpeg 时出错: {e}")
        


# === 主推理逻辑 ===
def run_vote_inference(qa_data_path, image_root_dir, frame_indices_dict_path, save_log_path):
    with open(qa_data_path, 'r') as f:
        qa_items = json.load(f)

    with open(frame_indices_dict_path, 'r') as f:
        frame_indices_list = json.load(f)

    
    # qa_items = qa_items[0:120]  # 1
    # qa_items = qa_items[120:240]  # 2
    qa_items = qa_items[240:360]  # 1
    # qa_items = qa_items[360:480]  # 2

    for idx, entry in enumerate(qa_items):
        video_id = os.path.splitext(os.path.basename(entry['video_path']))[0]
        all_indices = frame_indices_list[idx]
        total = len(all_indices)
        # class 1
        start = int(total * 0.5)
        end = int(total * 0.7)
        # class 2
        # start = int(total * 0.75)
        # end = int(total * 0.95)
        selected_indices = all_indices[start:end]
        question = entry['question']
        choices = entry['choices']
        answer = entry['answer']
        # print(f"\n[{idx+1}/{len(qa_items)}] Video: {video_id} | Question: {question}")
        for fid in selected_indices:
            frame_path = os.path.join(image_root_dir, f"{video_id}_q{idx}", f"{video_id}_frame_{fid}.jpg")
            os.makedirs(os.path.dirname(os.path.join(image_root_dir, f"{video_id}_q{idx}")), exist_ok=True)
            extract_frame(video_path=entry['video_path'], frame_index=fid, output_path=frame_path)

    # 推理循环
    log = []
    correct = 0


# === CLI ===
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--qa_data', type=str, required=True)
    # parser.add_argument('--image_dir', type=str, required=True, help="Root folder where per-video JPG folders exist")
    # parser.add_argument('--frames', type=str, required=True, help="JSON file with list of key frame indices per QA")
    # parser.add_argument('--save_log', type=str, required=True, help="Directory to save log JSON file")
    # args = parser.parse_args()

    # run_vote_inference(args.qa_data, args.image_dir, args.frames, args.save_log)
    QA = "/home/cxx/HWs/AKS/Mixed Video Pipeline/outputs/mixed_video_qa_randomized_4x_per_video.json"
    img = "/home/cxx/HWs/AKS/Mixed Video Pipeline/Testing/outputs/Shooting/class_1/jpgs"
    frames = "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Shooting/class_1/blip/optimized_frames/optimized_frames.json"
    save_log = "XXXX"
    run_vote_inference(QA, img, frames, save_log)