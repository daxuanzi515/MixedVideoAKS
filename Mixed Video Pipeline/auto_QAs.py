# video_to_qa_pipeline.py
# End-to-end pipeline: MP4 videos -> keyframes -> BLIP-2 description -> QA -> LongVideoBench JSON

import os
import json
import cv2
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# -------------------------------
# Step 1: Extract middle frame
# -------------------------------
def extract_middle_frame(video_path, output_dir):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    success, frame = cap.read()
    cap.release()

    if success:
        img_path = Path(output_dir) / (Path(video_path).stem + "_frame.jpg")
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(img_path)
        return img_path
    return None

# -------------------------------
# Step 2: Generate description using BLIP-2
# -------------------------------
def load_blip2_model():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto", torch_dtype=torch.float16)
    model.eval()
    return processor, model

def describe_image_with_blip2(img_path, processor, model):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# -------------------------------
# Step 3: Generate QA from description
# -------------------------------
def generate_qa_from_description(video_id, description):
    q = {
        "video_id": video_id,
        "question": f"What are the individuals doing in the scene where {description.lower()}",
        "question_wo_referring_query": f"What are the individuals doing {description[description.find('are') + 4:].lower()}",
        "candidates": [
            "They are walking calmly without any conflict.",
            "They are engaging in violent behavior.",
            "They are trying to board a train.",
            "They are helping each other move through the hallway."
        ],
        "correct_choice": 1,
        "position": [0, 30],
        "topic_category": "VD-Violent-Detection",
        "question_category": "S2E",
        "level": "L1-Perception",
        "id": f"{video_id}_0",
        "video_path": f"{video_id}.mp4",
        "subtitle_path": None,
        "duration_group": 60,
        "starting_timestamp_for_subtitles": 0,
        "duration": 30.0,
        "view_count": 0
    }
    return q

# -------------------------------
# Step 4: Run full pipeline
# -------------------------------
def process_videos_to_json(video_dir, output_json_path):
    processor, model = load_blip2_model()
    videos = sorted(Path(video_dir).glob("*.mp4"))
    frame_dir = Path(video_dir) / "frames"
    frame_dir.mkdir(exist_ok=True)

    all_qas = []

    for video in tqdm(videos):
        frame_path = extract_middle_frame(video, frame_dir)
        if not frame_path:
            print(f"❌ Failed to extract frame from {video.name}")
            continue
        description = describe_image_with_blip2(frame_path, processor, model)
        qa = generate_qa_from_description(video.stem, description)
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
