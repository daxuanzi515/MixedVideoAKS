import numpy as np
import json
import argparse
import os
from collections import deque

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='longvideobench')
    parser.add_argument('--extract_feature_model', type=str, default='blip')
    parser.add_argument('--score_path', type=str, default='achrives/outscores/longvideobench/blip/scores.json')
    parser.add_argument('--frame_path', type=str, default='achrives/outscores/longvideobench/blip/frames.json')
    parser.add_argument('--max_num_frames', type=int, default=64)
    parser.add_argument('--ratio', type=int, default=1)
    parser.add_argument('--t1', type=float, default=0.8)
    parser.add_argument('--t2', type=float, default=-100)
    parser.add_argument('--all_depth', type=int, default=5)
    parser.add_argument('--output_file', type=str, default='optimized_results')
    return parser.parse_args()

def select_segments(scores, frames, n, t1, t2, max_depth):
    queue = deque()
    queue.append((scores, frames, 0))  # (score_array, frame_list, depth)
    selected_segments = []

    while queue:
        s, f, d = queue.popleft()
        s = np.array(s)
        mean = s.mean()
        std = s.std()
        k = min(n, len(s))  # 防止 top-k 超出边界
        top_idx = np.argpartition(s, -k)[-k:]
        top_scores = s[top_idx]
        mean_diff = top_scores.mean() - mean

        print(f"[Depth {d}] Segment length = {len(s)} | Mean = {mean:.3f} | Std = {std:.3f} | MeanDiff = {mean_diff:.3f}")

        if mean_diff > t1 and std > t2:
            print(f"  ✅ Accept segment at depth {d}")
            selected_segments.append((s, f, d))
        elif d < max_depth:
            mid = len(s) // 2
            if mid > 0:
                queue.append((s[:mid], f[:mid], d + 1))
                queue.append((s[mid:], f[mid:], d + 1))
            else:
                print(f"  ⚠ Too short to split. Accept anyway.")
                selected_segments.append((s, f, d))
        else:
            print(f"  ❌ Reached max depth. Accept anyway.")
            selected_segments.append((s, f, d))
    return selected_segments

def main(args):
    with open(args.score_path) as f:
        itm_outs = json.load(f)
    with open(args.frame_path) as f:
        fn_outs = json.load(f)

    output_path = os.path.join(args.output_file, args.dataset_name, args.extract_feature_model)
    os.makedirs(output_path, exist_ok=True)

    final_outputs = []

    for idx, (scores, frames) in enumerate(zip(itm_outs, fn_outs)):
        print(f"\n>>> Processing video {idx}: {len(scores)} raw frames")
        scores = scores[::args.ratio]
        frames = frames[::args.ratio]

        scores = np.array(scores)
        if len(scores) >= 8:
            if scores.max() > scores.min():
                scores = (scores - scores.min()) / (scores.max() - scores.min())

            segments = select_segments(scores, frames, args.max_num_frames, args.t1, args.t2, args.all_depth)

            selected = []
            for i, (seg_scores, seg_frames, depth) in enumerate(segments):
                k = min(args.max_num_frames // (2 ** depth), len(seg_scores))
                top_k_idx = np.argpartition(seg_scores, -k)[-k:]
                selected_frames = [seg_frames[i] for i in top_k_idx]
                print(f"  Segment {i} at depth {depth} → selected {len(selected_frames)} frames")
                selected.extend(selected_frames)

            final_outputs.append(sorted(selected))
        else:
            print("⚠ Not enough frames, using all.")
            final_outputs.append(frames)

    output_json = os.path.join(output_path, 'selected_frames.json')
    with open(output_json, 'w') as f:
        json.dump(final_outputs, f)
    print(f"\n✅ Finished. Selected frames saved to: {output_json}")

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
