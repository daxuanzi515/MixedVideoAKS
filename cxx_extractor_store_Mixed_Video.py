import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from transformers import CLIPProcessor, CLIPModel

import json
from decord import VideoReader
from decord import cpu
import numpy as np
import os
import pickle
import argparse
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract Video Feature')
    parser.add_argument('--dataset_name', type=str, default='Fighting', help='support longvideobench and videomme')
    parser.add_argument('--dataset_path', type=str, default='/home/cxx/HWs/AKS/Mixed Video Pipeline/outputs/Fighting', help='your path of the dataset')
    parser.add_argument('--extract_feature_model', type=str, default='blip', help='blip/clip')
    parser.add_argument('--output_file', type=str, default='/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things', help='path of output scores and frames')
    parser.add_argument('--classes', type=str, default='class_1')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--min_videos', type=int, default=None, help='Only process this many unique videos (for debug)')
    parser.add_argument('--max_videos', type=int, default=None, help='Only process this many unique videos (for debug)')
    return parser.parse_args()

def main(args):
    root_path = f"/home/cxx/HWs/AKS/Mixed Video Pipeline/outputs/"
    if args.dataset_name in ["Fighting", "Shooting"]:
        label_path = os.path.join(root_path, 'mixed_video_qa_randomized_4x_per_video.json')
        video_path = os.path.join(args.dataset_path, args.classes)
    else:
        raise ValueError("dataset_name must be: Fighting or Shooting")

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found at {label_path}")
    with open(label_path, 'r') as f:
        datas = json.load(f)

    if args.max_videos is not None and args.min_videos is not None:
        datas = datas[args.min_videos:args.max_videos]
        
    device = args.device
    if args.extract_feature_model == 'blip':
        model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)
    elif args.extract_feature_model == 'clip':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model.to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise ValueError("extract_feature_model must be: blip or clip")

    out_score_path = os.path.join(args.output_file, args.dataset_name, args.classes, args.extract_feature_model)
    os.makedirs(out_score_path, exist_ok=True)
    score_dir = os.path.join(out_score_path, "scores")
    frame_dir = os.path.join(out_score_path, "frames")
    embedding_dir = os.path.join(out_score_path, "embeddings")
    os.makedirs(score_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)

    failed_videos = []
    index_map = []

    expected_prefix = os.path.abspath(os.path.join(args.dataset_path, args.classes))

    for idx, data in enumerate(datas):
        start_time = time.time()
        try:
            text = data['question']
            video_file = data["video_path"]
            video_id = os.path.splitext(os.path.basename(video_file))[0]

            # ✅ 仅保留当前 class 内的视频
            if not os.path.abspath(video_file).startswith(expected_prefix):
                print(f"[{idx + 1}] ⚠️ Skipped (outside class): {video_file}")
                failed_videos.append({
                    "index": idx,
                    "video_id": video_id,
                    "video_path": video_file,
                    "question": text,
                    "error": f"Video path not under target class ({args.classes})"
                })
                continue

            # ✅ 若视频文件不存在，也跳过
            if not os.path.exists(video_file):
                print(f"[{idx + 1}] ⚠️ Skipped (file not found): {video_file}")
                failed_videos.append({
                    "index": idx,
                    "video_id": video_id,
                    "video_path": video_file,
                    "question": text,
                    "error": "Video file not found"
                })
                continue

            file_prefix = f"{video_id}_q{idx}"
            print(f"\n[{idx + 1}/{len(datas)}] Processing video: {video_file} | question: {text[:40]}...")

            vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
            fps = vr.get_avg_fps()
            frame_nums = int(len(vr) / int(fps))

            score = []
            frame_num = []
            embedding = []

            if args.extract_feature_model == 'blip':
                txt = text_processors["eval"](text)
                for j in range(frame_nums):
                    raw_image = np.array(vr[j * int(fps)])
                    raw_image = Image.fromarray(raw_image)
                    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        blip_output, blip_embedding = model({"image": img, "text_input": txt}, match_head="itm-e")
                    blip_scores = torch.nn.functional.softmax(blip_output, dim=1)
                    embedding.append(blip_embedding.cpu())
                    score.append(blip_scores[:, 1].item())
                    frame_num.append(j * int(fps))

            elif args.extract_feature_model == 'clip':
                inputs_text = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
                text_features = model.get_text_features(**inputs_text)
                for j in range(frame_nums):
                    raw_image = np.array(vr[j * int(fps)])
                    raw_image = Image.fromarray(raw_image)
                    inputs_image = processor(images=raw_image, return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        image_features = model.get_image_features(**inputs_image)
                    clip_score = torch.nn.functional.cosine_similarity(text_features, image_features)
                    embedding.append(image_features.cpu())
                    score.append(clip_score.item())
                    frame_num.append(j * int(fps))

            with open(os.path.join(score_dir, f"{file_prefix}.json"), "w") as f:
                json.dump(score, f)
            with open(os.path.join(frame_dir, f"{file_prefix}.json"), "w") as f:
                json.dump(frame_num, f)
            with open(os.path.join(embedding_dir, f"{file_prefix}.pkl"), "wb") as f:
                pickle.dump(embedding, f)

            index_map.append({
                "index": idx,
                "file_prefix": file_prefix,
                "video_id": video_id,
                "video_path": video_file,
                "question": text,
                "choices": data.get("choices", []),
                "answer": data.get("answer", None)
            })

            print(f"[{file_prefix}] ✅ Done in {time.time() - start_time:.2f} seconds.")

        except Exception as e:
            error_info = f"{type(e).__name__}: {str(e)}"
            print(f"[{idx + 1}/{len(datas)}] ❌ Failed to process: {error_info}")
            failed_videos.append({
                "index": idx,
                "video_id": video_id,
                "question": text,
                "error": error_info
            })
            continue

    with open(os.path.join(out_score_path, 'index_map.json'), 'w') as f:
        json.dump(index_map, f, indent=2)

    if failed_videos:
        with open(os.path.join(out_score_path, 'failed_videos.json'), 'w') as f:
            json.dump(failed_videos, f, indent=2)
        print(f"\n⚠️ {len(failed_videos)} videos failed. See failed_videos.json.")
    else:
        print("\n✅ All questions processed successfully.")



if __name__ == '__main__':
    args = parse_arguments()
    main(args)

