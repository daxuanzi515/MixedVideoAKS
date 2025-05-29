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
    parser.add_argument('--dataset_name', type=str, default='longvideobench', help='support longvideobench and videomme')
    parser.add_argument('--dataset_path', type=str, default='./datasets/longvideobench', help='your path of the dataset')
    parser.add_argument('--extract_feature_model', type=str, default='blip', help='blip/clip')
    parser.add_argument('--output_file', type=str, default='./outscores', help='path of output scores and frames')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_videos', type=int, default=None, help='Only process this many unique videos (for debug)')
    return parser.parse_args()

def main(args):
    if args.dataset_name == "longvideobench":
        label_path = os.path.join(args.dataset_path, 'lvb_val.json')
        video_path = os.path.join(args.dataset_path, 'videos')
    elif args.dataset_name == "videomme":
        label_path = os.path.join(args.dataset_path, 'videomme.json')
        video_path = os.path.join(args.dataset_path, 'data')
    else:
        raise ValueError("dataset_name must be: longvideobench or videomme")

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found at {label_path}")
    with open(label_path, 'r') as f:
        datas = json.load(f)

    # Filter unique videos
    seen_video_ids = set()
    filtered_datas = []
    for d in datas:
        video_file = d["video_path"] if args.dataset_name == 'longvideobench' else d["videoID"] + ".mp4"
        video_id = os.path.splitext(os.path.basename(video_file))[0]
        if video_id not in seen_video_ids:
            filtered_datas.append(d)
            seen_video_ids.add(video_id)
    if args.max_videos is not None:
        filtered_datas = filtered_datas[:args.max_videos]
    datas = filtered_datas

    device = args.device
    if args.extract_feature_model == 'blip':
        model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)
    elif args.extract_feature_model == 'clip':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model.to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise ValueError("extract_feature_model must be: blip or clip")

    out_score_path = os.path.join(args.output_file, args.dataset_name, args.extract_feature_model)
    os.makedirs(out_score_path, exist_ok=True)
    score_dir = os.path.join(out_score_path, "scores")
    frame_dir = os.path.join(out_score_path, "frames")
    embedding_dir = os.path.join(out_score_path, "embeddings")
    os.makedirs(score_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)

    failed_videos = []

    for idx, data in enumerate(datas):
        start_time = time.time()
        try:
            text = data['question']
            video_file = data["video_path"] if args.dataset_name == 'longvideobench' else data["videoID"] + ".mp4"
            video = os.path.join(video_path, video_file)
            video_id = os.path.splitext(os.path.basename(video_file))[0]

            print(f"\n[{idx + 1}/{len(datas)}] Processing video: {video_id} ...")

            vr = VideoReader(video, ctx=cpu(0), num_threads=1)
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
                    blip_embedding = blip_embedding.cpu()
                    embedding.append(blip_embedding)
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
                    clip_embedding = image_features.cpu()
                    embedding.append(clip_embedding)
                    score.append(clip_score.item())
                    frame_num.append(j * int(fps))

            with open(os.path.join(score_dir, f"{video_id}.json"), "w") as f:
                json.dump(score, f)
            with open(os.path.join(frame_dir, f"{video_id}.json"), "w") as f:
                json.dump(frame_num, f)
            with open(os.path.join(embedding_dir, f"{video_id}.pkl"), "wb") as f:
                pickle.dump(embedding, f)

            print(f"[{video_id}] ✅ Done in {time.time() - start_time:.2f} seconds.")

        except Exception as e:
            error_info = f"{type(e).__name__}: {str(e)}"
            print(f"[{idx + 1}/{len(datas)}] ❌ Failed to process video {video_id}: {error_info}")
            failed_videos.append({
                "video_id": video_id,
                "index": idx,
                "error": error_info
            })
            continue

    failed_log_path = os.path.join(out_score_path, 'failed_videos.json')
    if failed_videos:
        with open(failed_log_path, 'w') as f:
            json.dump(failed_videos, f, indent=2)
        print(f"\n⚠️ {len(failed_videos)} videos failed. Details saved to {failed_log_path}")
    else:
        print("\n✅ All videos processed successfully.")

if __name__ == '__main__':
    args = parse_arguments()
    main(args)

