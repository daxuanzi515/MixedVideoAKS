# video_to_qa_pipeline.py
# End-to-end pipeline: MP4 videos -> keyframes -> BLIP-2 description -> QA -> LongVideoBench JSON

import os
import json
import cv2
import random
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# -------------------------------
# Step 1: Extract random frame and timestamp (with fallback on content)
# -------------------------------
def extract_random_frame(video_path, output_dir):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total_frames <= 0 or fps <= 0:
        return None, None, None, None

    attempts = 5
    for _ in range(attempts):
        rand_frame = random.randint(total_frames // 4, 3 * total_frames // 4)
        rand_time = rand_frame / fps
        cap.set(cv2.CAP_PROP_POS_FRAMES, rand_frame)
        success, frame = cap.read()
        if success:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.std() < 5:
                continue  # skip blank or nearly black frames
            img_path = Path(output_dir) / (Path(video_path).stem + f"_frame_{rand_frame}.jpg")
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(img_path)
            cap.release()
            return img_path, rand_time, fps, total_frames
    cap.release()
    return None, None, None, None

# -------------------------------
# Step 2: Generate description using BLIP-2 (fallback-safe)
# -------------------------------
def load_blip2_model():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16
    ).to("cuda")
    model.eval()
    return processor, model

def describe_image_with_blip2(img_path, processor, model):
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(output[0], skip_special_tokens=True).strip()
        if not caption or len(caption.strip()) < 5:
            fallbacks = [
                "a group of people are standing close",
                "two people are interacting with some tension",
                "someone seems to be raising their hand",
                "a crowd is present but not clearly visible",
                "individuals are partially obstructed"
            ]
            return random.choice(fallbacks)
        return caption
    except Exception as e:
        print(f"⚠️ BLIP-2 description failed for {img_path.name}: {e}")
        return "people are interacting with some tension"

# -------------------------------
# Step 3: Generate QA from description (LongVideoBench-style)
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
    candidates = [
        f"They are {a}." for a in base_actions[:4]
    ]

    violent_keywords = ["violent", "pushing", "arguing"]
    correct_index_list = [i for i, s in enumerate(candidates) if any(k in s for k in violent_keywords)]

    if not correct_index_list:
        candidates[0] = "They are engaging in violent behavior."
        correct_index_list = [0]
    elif len(correct_index_list) > 2:
        correct_index_list = correct_index_list[:2]

    q = {
        "video_id": video_id,
        "question": f"What are the individuals doing in the scene where {description.lower().rstrip('.')}?",
        "question_wo_referring_query": f"What are the individuals doing {description.lower().rstrip('.')}?",
        "candidates": candidates,
        "correct_choice": sorted(correct_index_list),
        "position": position_frames,
        "topic_category": "VD-Violent-Detection",
        "question_category": "S2E",
        "level": "L1-Perception",
        "id": f"{video_id}_0",
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
    processor, model = load_blip2_model()
    videos = sorted(Path(video_dir).glob("*.mp4"))[:5]  # only first 5 videos
    frame_dir = Path(video_dir) / "frames"
    frame_dir.mkdir(exist_ok=True)

    all_qas = []

    for video in tqdm(videos):
        frame_path, rand_time, fps, total_frames = extract_random_frame(video, frame_dir)
        if not frame_path:
            print(f"❌ Failed to extract frame from {video.name}")
            continue
        description = describe_image_with_blip2(frame_path, processor, model)
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
    output_json = "Mixed Video Pipeline/val.json"
    process_videos_to_json(video_dir, output_json)
