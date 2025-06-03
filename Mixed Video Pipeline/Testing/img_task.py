import argparse
import os
import json
from PIL import Image
from collections import Counter
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# === 初始化模型 ===
MODEL_PATH = "/home/cxx/HWs/AKS/checkpoints/MiniCPM-V-2_6"
llm = LLM(model=MODEL_PATH, gpu_memory_utilization=1, trust_remote_code=True, max_model_len=2048)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
stop_tokens = ['<|im_end|>', '<|endoftext|>']
stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
sampling_params = SamplingParams(stop_token_ids=stop_token_ids, temperature=0, max_tokens=1024)

# === 输入文件路径 ===
qa_json_path = "/home/cxx/HWs/AKS/Mixed Video Pipeline/outputs/mixed_video_qa_randomized_4x_per_video.json"
frame_dir = "/home/cxx/HWs/AKS/Mixed Video Pipeline/Testing/outputs/Fighting/class_1/jpgs"  # 所有帧名格式：Test_0_Mixed_frame_192.jpg
frame_indices_dict_path = "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Fighting/class_1/blip/optimized_frames/optimized_frames.json"  # 每条 QA 的关键帧 index list（按顺序）

def build_prompt(question, choices):
    choice_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    return (
        "(<image>./</image>)\n"
        "You are analyzing a scene from a video. Carefully observe the image and select the most appropriate answer from the choices below.\n\n"
        f"Question:\n{question}\n\nChoices:\n{choice_str}\n\n"
        "Answer (choose only one option from above, e.g., \"A. Shooting\"):"
    )

def run_vote_inference(qa_data_path, image_root_dir, frame_indices_dict_path):
    MODEL_PATH = "/home/cxx/HWs/AKS/checkpoints/MiniCPM-V-2_6"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    llm = LLM(model=MODEL_PATH, gpu_memory_utilization=1, trust_remote_code=True, max_model_len=2048)

    with open(qa_data_path, 'r') as f:
        qa_items = json.load(f)

    with open(frame_indices_dict_path, 'r') as f:
        frame_indices_list = json.load(f)

    log = []
    correct = 0

    for idx, entry in enumerate(qa_items):
        video_id = os.path.splitext(os.path.basename(entry['video_path']))[0]
        frame_indices = frame_indices_list[idx]
        question = entry['question']
        choices = entry['choices']
        answer = entry['answer']

        all_outputs = []
        for fid in frame_indices:
            image_name = f"{video_id}_frame_{fid}.jpg"
            image_path = os.path.join(image_root_dir, video_id, image_name)

            if not os.path.exists(image_path):
                print(f"⚠️ Skipping missing image: {image_path}")
                continue

            image = Image.open(image_path).convert("RGB")
            messages = [{"role": "user", "content": build_prompt(question, choices)}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            sampling_params = SamplingParams(
                stop_token_ids=[tokenizer.convert_tokens_to_ids(i) for i in ['<|im_end|>', '<|endoftext|>']],
                temperature=0,
                max_tokens=1024,
                best_of=1)

            outputs = llm.generate({
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image
                }
            }, sampling_params=sampling_params)

            out_text = outputs[0].outputs[0].text.strip()
            all_outputs.append(out_text)

        # 投票
        counter = Counter()
        for output in all_outputs:
            for choice in choices:
                if choice.lower() in output.lower():
                    counter[choice] += 1
                    break
        voted = counter.most_common(1)[0][0] if counter else "None"

        result = {
            "video_id": video_id,
            "question": question,
            "choices": choices,
            "gt_answer": answer,
            "pred_answer": voted,
            "raw_outputs": all_outputs
        }
        log.append(result)
        if voted == answer:
            correct += 1

        print(f"[{idx+1}/{len(qa_items)}] ✅ Pred: {voted} | GT: {answer}")

    acc = correct / len(qa_items)
    print(f"\n✅ Overall Accuracy: {acc:.2%} ({correct}/{len(qa_items)})")

    with open("voting_results_log.json", "w") as f:
        json.dump(log, f, indent=2)

# CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa_data', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True, help="Root folder where per-video JPG folders exist")
    parser.add_argument('--frames', type=str, required=True, help="JSON file with list of key frame indices per QA")
    args = parser.parse_args()

    run_vote_inference(args.qa_data, args.image_dir, args.optimized_frames)


