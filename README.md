# README

In short, my record and testing tips here, depending on my environment: Ubuntu 20.04, 3090Ti, CUDA 12.2, GPU 24GB, RAM 61GB, Python 3.9 & 3.10.

## Requirements Download
### Features Extractor
Just follow the commands:
```bash
conda create -n SeViLA python=3.9
conda activate SeViLA
git clone https://github.com/Yui010206/SeViLA.git
cd SeViLA
pip install -e .
pip install numpy==1.24.4
pip install spacy
```
Usage of SeViLA:
```bash
# only extract features in 3 videos of Fighting/class_1
python cxx_extractor_store_Mixed_Video.py --dataset_name Fighting --dataset_path "/home/cxx/HWs/AKS/Mixed Video Pipeline/outputs/Fighting" --extract_feature_model blip --output_file "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things" --classes class_1 --min_videos 0 --max_videos 3

# test exclusion
python cxx_extractor_store_Mixed_Video.py --dataset_name Fighting --dataset_path "/home/cxx/HWs/AKS/Mixed Video Pipeline/outputs/Fighting" --extract_feature_model blip --output_file "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things" --classes class_1 --min_videos 120 --max_videos 125

# orignal select 
python frame_select.py \
  --dataset_name Fighting \
  --extract_feature_model blip \
  --score_path "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Fighting/class_1/blip/scores.json" \
  --frame_path "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Fighting/class_1/blip/frames.json" \
  --output_file "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Fighting/class_1/blip/selected_frames"

python frame_select.py \
  --dataset_name Fighting \
  --extract_feature_model blip \
  --score_path "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Fighting/class_2/blip/scores.json" \
  --frame_path "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Fighting/class_2/blip/frames.json" \
  --output_file "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Fighting/class_2/blip/selected_frames"

python frame_select.py \
  --dataset_name Fighting \
  --extract_feature_model blip \
  --score_path "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Shooting/class_1/blip/scores.json" \
  --frame_path "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Shooting/class_1/blip/frames.json" \
  --output_file "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Shooting/class_1/blip/selected_frames"

python frame_select.py \
  --dataset_name Fighting \
  --extract_feature_model blip \
  --score_path "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Shooting/class_2/blip/scores.json" \
  --frame_path "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Shooting/class_2/blip/frames.json" \
  --output_file "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Shooting/class_2/blip/selected_frames"

# optimized
python optimized_select_frame.py \
  --dataset_name Fighting \
  --extract_feature_model blip \
  --score_path "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Fighting/class_1/blip/scores.json" \
  --frame_path "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Fighting/class_1/blip/frames.json" \
  --output_file "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Fighting/class_1/blip/optimized_frames"

python optimized_select_frame.py \
  --dataset_name Fighting \
  --extract_feature_model blip \
  --score_path "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Fighting/class_2/blip/scores.json" \
  --frame_path "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Fighting/class_2/blip/frames.json" \
  --output_file "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Fighting/class_2/blip/optimized_frames"

python optimized_select_frame.py \
  --dataset_name Fighting \
  --extract_feature_model blip \
  --score_path "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Shooting/class_1/blip/scores.json" \
  --frame_path "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Shooting/class_1/blip/frames.json" \
  --output_file "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Shooting/class_1/blip/optimized_frames"

python optimized_select_frame.py \
  --dataset_name Fighting \
  --extract_feature_model blip \
  --score_path "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Shooting/class_2/blip/scores.json" \
  --frame_path "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Shooting/class_2/blip/frames.json" \
  --output_file "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Shooting/class_2/blip/optimized_frames"

python img_task.py \
  --qa_data "/home/cxx/HWs/AKS/Mixed Video Pipeline/outputs/mixed_video_qa_randomized_4x_per_video.json" \
  --image_dir "/home/cxx/HWs/AKS/Mixed Video Pipeline/Testing/outputs/Fighting/class_1/jpgs" \
  --frames "/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/Fighting/class_1/blip/optimized_frames/optimized_frames.json"

```


### MLLMs Deployment
Pay attention to the version of the lmms-eval. The latest version may not work in python 3.9. 
Please use the following commands to install the environment:

```bash 
conda create -n AKS python=3.9
conda activate AKS
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
pip install -e ".[train]"
cd ..
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval -b v0.2.4
cd lmms-eval
pip install -e .
```
## Our Contributed Tasks
Our github repository: [MixedVideoAKS](https://github.com/daxuanzi515/MixedVideoAKS/tree/cxx)

Mixed Video Dataset: [baidu.pan](https://pan.baidu.com/s/1BsIIcddJqneNqLrZFGS_kA?pwd=7777)

Paper Link: [overleaf.cs240_group06](https://www.overleaf.com/read/qnyxnhfgfwpk#f0ea4a)

- **Synthesize Mixed Datasets.** Mixed Datasets are based on [XD-Violence](https://github.com/XD-DENG/XD-Violence) and [longvideobench](https://github.com/longvideobench/longvideobench). Then, we randomly generate QAs by AI and manually label keyframes. Finally, we use the labeled data to synthesize the mixed datasets.
- **Optimize the Select Frame Algorithm.** We optimize the original algorithm, and discuss its time complexity and correctness. Specifically, we have done the following things: 
  - **Modular and Depth-Aware Structure:** Enhanced algorithm with depth-controlled splitting and robust boundary checks.
  - **Edge Case Handling:** Improved safety for edge cases (e.g., $( K > |s| )$) to prevent runtime errors.
  - **Efficient Top-k Selection:** Reduced time complexity from $ \mathcal{O}(d \cdot n \log n) $ to $ \mathcal{O}(n \log d) $ by using partial top-k retrieval.
  - **Consistent Output:** Maintained consistent output length by enforcing a user-defined maximum $ K $ and adapting segment granularity based on content variance.
- **Apply Light-weight model for Testing**. We use the MiniCPM-V-2.6 model to test the select frame algorithm. We have done the following things:
  - **Test on KeyFrame Image.** We test the algorithm on a single image and compare the results with the original algorithm.
  - **Test on Mixed Videos.** We test the algorithm on the mixed dataset and compare the results with the original algorithm.

### Mixed Video Pipeline
Details see in [Mixed Video Pipeline](Mixed%20Video%20Pipeline/README.md).
### MiniCPM-V-2.6 Testings
Details see in [MiniCPM-V-2.6 Testings](Mixed%20Video%20Pipeline/Testing/MiniCPM-2.6.md).

