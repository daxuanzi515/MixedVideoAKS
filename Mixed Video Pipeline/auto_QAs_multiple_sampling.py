# video_to_qa_pipeline.py
# End-to-end pipeline: MP4 videos -> keyframes -> BLIP-like model description -> QA -> LongVideoBench JSON

import os
import json
import cv2
import random
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# -------------------------------
# Step 1: Extract multiple *consecutive* frames per video (more context)
# -------------------------------
def extract_multiple_frame_sets(video_path, output_dir, num_sets=3, frames_per_set=3, interval=1):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total_frames <= 0 or fps <= 0:
        return []

    extracted = []
    for _ in range(num_sets):
        base_frame = random.randint(total_frames // 6, 5 * total_frames // 6 - frames_per_set * interval)
        frame_set = []
        for i in range(frames_per_set):
            frame_idx = base_frame + i * interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if not success:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.std() < 5:
                break
            img_path = Path(output_dir) / (Path(video_path).stem + f"_frame_{frame_idx}.jpg")
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(img_path)
            frame_set.append(img_path)
        if len(frame_set) == frames_per_set:
            extracted.append((frame_set, base_frame / fps, fps, total_frames))
    cap.release()
    return extracted

# -------------------------------
# Step 2: Generate aggregated description using InternVL-Chat (BLIP alternative)
# -------------------------------
def load_multimodal_model():
    processor = AutoProcessor.from_pretrained(
        "OpenGVLab/InternVL-Chat-V1-5", 
        trust_remote_code=True  # ✅ 添加这行
    )
    model = AutoModelForCausalLM.from_pretrained(
        "OpenGVLab/InternVL-Chat-V1-5",
        torch_dtype=torch.float16,
        trust_remote_code=True  # ✅ 已有
    ).to("cuda").eval()
    return processor, model


def describe_frameset_with_internvl(frame_set, processor, model):
    from torchvision import transforms
    images = [Image.open(p).convert("RGB") for p in frame_set]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    tensor_images = torch.stack([transform(img) for img in images]).to(model.device)
    query = "What is happening in these frames?"
    inputs = processor(text=query, images=tensor_images, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return caption

# -------------------------------
# Step 3: Generate QA from aggregated description (LongVideoBench-style)
# -------------------------------
def generate_qa_from_description(video_id, description, rand_time, fps, total_frames):
    rand_frame = int(rand_time * fps)
    position_frames = sorted(list(set([
        max(0, rand_frame - 60),
        rand_frame,
        min(total_frames - 1, rand_frame + 60)
    ])))

    base_actions = [
        "engaging in violent behavior",
        "walking calmly without any conflict",
        "running away from someone",
        "arguing with gestures",
        "sitting together peacefully",
        "trying to board a train",
        "playing with each other",
        "pushing one another",
        "looking around vigilantly",
        "helping someone stand up"
    ]

    random.shuffle(base_actions)
    candidates = [f"They are {a}." for a in base_actions[:4]]

    violent_keywords = ["violent", "pushing", "arguing"]
    correct_index_list = [i for i, s in enumerate(candidates) if any(k in s for k in violent_keywords)]

    if not correct_index_list:
        candidates[0] = "They are engaging in violent behavior."
        correct_index_list = [0]
    elif len(correct_index_list) > 2:
        correct_index_list = correct_index_list[:2]

    q = {
        "video_id": video_id,
        "question": f"What are the individuals doing in the scene where {description.lower().rstrip('.')}",
        "question_wo_referring_query": f"What are the individuals doing {description.lower().rstrip('.')}",
        "candidates": candidates,
        "correct_choice": sorted(correct_index_list),
        "position": position_frames,
        "topic_category": "VD-Violent-Detection",
        "question_category": "S2E",
        "level": "L1-Perception",
        "id": f"{video_id}_{rand_frame}",
        "video_path": f"{video_id}.mp4",
        "subtitle_path": None,
        "duration_group": 60,
        "starting_timestamp_for_subtitles": 0,
        "duration": round(total_frames / fps, 2),
        "view_count": 0
    }
    return q

# -------------------------------
# Step 4: Run full pipeline (testing on 5 videos only)
# -------------------------------
def process_videos_to_json(video_dir, output_json_path):
    processor, model = load_multimodal_model()
    videos = sorted(Path(video_dir).glob("*.mp4"))[:5]  # only first 5 videos
    frame_dir = Path(video_dir) / "frames"
    frame_dir.mkdir(exist_ok=True)

    all_qas = []

    for video in tqdm(videos):
        extracted_sets = extract_multiple_frame_sets(video, frame_dir, num_sets=3, frames_per_set=3)
        if not extracted_sets:
            print(f"❌ Failed to extract frames from {video.name}")
            continue
        for frame_set, rand_time, fps, total_frames in extracted_sets:
            description = describe_frameset_with_internvl(frame_set, processor, model)
            print(f"🧠 {video.stem}: {description}")
            qa = generate_qa_from_description(video.stem, description, rand_time, fps, total_frames)
            all_qas.append(qa)

    with open(output_json_path, "w") as f:
        json.dump(all_qas, f, indent=2)
    print(f"✅ Done! QA saved to {output_json_path}")

# -------------------------------
# Usage Example
# -------------------------------
if __name__ == "__main__":
    video_dir = "datasets/XD_violence/ours/Fighting"  # folder with .mp4 files
    output_json = "Mixed Video Pipeline/new_val.json"
    process_videos_to_json(video_dir, output_json)
