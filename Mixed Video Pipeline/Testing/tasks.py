# from modelscope import snapshot_download

# # 指定下载路径
# model_dir = snapshot_download('linglingdan/MiniCPM-V_2_6_awq_int4', cache_dir='/home/cxx/HWs/AKS/checkpoints')

# # 打印下载路径
# print(f"模型下载路径: {model_dir}")


import os
import sys
# import torch
# import numpy as np
# print("PyTorch版本:", torch.__version__)
# print("CUDA是否可用:", torch.cuda.is_available())
# print("CUDA版本:", torch.version.cuda)
# print("GPU型号:", torch.cuda.get_device_name(0))
# print("numpy版本:", np.__version__)

# import numpy as np
# print("Numpy version:", np.__version__)
# print("Array type:", type(np.array([1.0])))
# print("Is instance of np.ndarray:", isinstance(np.array([1.0]), np.ndarray))


from PIL import Image
from transformers import AutoTokenizer
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

import vllm
from vllm import LLM, SamplingParams

# 图像文件路径列表
IMAGES = [
    "/home/cxx/HWs/AKS/datasets/img/bto.jpg",  # 本地图片路径
]

# 模型名称或路径
MODEL_NAME = "/home/cxx/HWs/AKS/checkpoints/MiniCPM-V-2_6" 

# 打开并转换图像
image = Image.open(IMAGES[0]).convert("RGB")

# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 初始化语言模型
llm = LLM(model=MODEL_NAME,
           gpu_memory_utilization=1,  # 使用全部GPU内存
           trust_remote_code=True,
           max_model_len=2048)  # 根据内存状况可调整此值

# 构建对话消息
messages = [{'role': 'user', 'content': '(<image>./</image>)\n' + '请描述这张图片'}]

# 应用对话模板到消息
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 设置停止符ID
# 2.0
# stop_token_ids = [tokenizer.eos_id]
# 2.5
#stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]
# 2.6 
stop_tokens = ['<|im_end|>', '<|endoftext|>']
stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

# 设置生成参数
sampling_params = SamplingParams(
    stop_token_ids=stop_token_ids,
    # temperature=0.7,
    # top_p=0.8,
    # top_k=100,
    # seed=3472,
    max_tokens=1024,
    # min_tokens=150,
    temperature=0,
    # use_beam_search=True, # none
    # length_penalty=1.2,
    best_of=1) # greedy must be 1)

# 获取模型输出
outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {
        "image": image
    }
}, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)

# import os
# import sys
# import types
# import numpy as np
# import torch
# from transformers import AutoTokenizer
# from decord import VideoReader, cpu
# from PIL import Image
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# sys.path.append(project_root)
# from checkpoints.vllm.vllm import LLM, SamplingParams


# # 进行图片推理
# MAX_NUM_FRAMES = 16
# def encode_video(filepath):
#     def uniform_sample(l, n):
#         gap = len(l) / n
#         idxs = [int(i * gap + gap / 2) for i in range(n)]
#         return [l[i] for i in idxs]
#     vr = VideoReader(filepath, ctx=cpu(0))
#     sample_fps = round(vr.get_avg_fps() / 1)  # FPS
#     frame_idx = [i for i in range(0, len(vr), sample_fps)]
#     if len(frame_idx)>MAX_NUM_FRAMES:
#         frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
#     video = vr.get_batch(frame_idx).asnumpy()
#     video = [Image.fromarray(v.astype('uint8')) for v in video]
#     return video

# MODEL_NAME = "/home/cxx/HWs/AKS/checkpoints/MiniCPM-V-2_6" 
# # openbmb/MiniCPM-V-2_6: auto download from huggingface
# # local path: checkpoints/MiniCPM-V-2_6
# llm = LLM(
#     model=MODEL_NAME,
#     gpu_memory_utilization=0.95,
#     max_model_len=2048,
#     trust_remote_code=True,
# )

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# stop_tokens = ['<|im_end|>', '<|endoftext|>']
# stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

# video_path = f"/home/cxx/HWs/AKS/datasets/XD_violence/ours/Shooting/Test_0_Shooting.mp4"
# frames = encode_video(video_path)
# messages = [{
#     "role":
#     "user",
#     "content":
#     "".join(["(<image>./</image>)"] * len(frames)) + "\nPlease describe this video."
# }]

# prompt = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )

# sampling_params = SamplingParams(
#     stop_token_ids=stop_token_ids, 
#     use_beam_search=False,
#     temperature=0.7,
#     top_p=0.8,
#     top_k=100, 
#     max_tokens=512
# )

# outputs = llm.generate({
#     "prompt": prompt,
#     "multi_modal_data": {
#         "image": {
#             "images": frames,
#             "use_image_id": False,
#             "max_slice_nums": 1 if len(frames) > 16 else 2
#         }
#     }
# }, sampling_params=sampling_params)

# print(outputs)