# My MiniCPM-2.6 Deployment
## Step by Step: Quick Start
Create a new conda environment:
```bash
conda create -n AKS python=3.9
conda activate AKS
```

Basic Download: with vllm

MiniCPM-2.6: 
```bash
git clone https://huggingface.co/openbmb/MiniCPM-V-2_6
```

awq model: 
```bash
git clone https://www.modelscope.cn/models/linglingdan/MiniCPM-V_2_6_awq_int4
# 安装Autoawq的分支，已经提了pr，等官方合并
git clone https://github.com/LDLINGLINGLING/AutoAWQ.git
cd AutoAWQ
git checkout minicpmv2.6
pip install -e .
```
Install vllm:
```bash
pip install vllm==0.5.4
```
Support multiple-images and videos input:
Pay attention to GPU configuration, you need to install `cuda-toolkit` >= 11.6, and `cmake` >= 3.20.0. in your virtual environment.
Then you should check your envrionment variable like `CUDA_HOME` is available or not.
Relaunch your runfile of cuda, mine is `cuda_12.2.0`.

Official link:https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run
```
Space for Dirver and Select CUDA Install.
After installation, you can check your `nvcc` version.
```bash
nvcc --version
```
Set Specific CUDA version to your bashrc or zshrc:
```bash
sudo gedit ~/.bashrc
```
Add these lines to the end of the file:
```bash
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```
Refresh your bashrc or zshrc:
```bash
source ~/.bashrc
```

Next, use this as must:
```bash
conda create -n vllm python=3.10
conda activate vllm
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install vllm
```


Check your package version:
```python
import torch
import numpy as np
print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
print("CUDA版本:", torch.version.cuda)
print("GPU型号:", torch.cuda.get_device_name(0))
print("numpy版本:", np.__version__)
# PyTorch版本: 2.5.1
# CUDA是否可用: True
# CUDA版本: 11.8
# GPU型号: NVIDIA GeForce RTX 3090
# numpy版本: 1.26.4
```



### Codes Demo
- Demo Code For Video
```python
from transformers import AutoTokenizer
from decord import VideoReader, cpu
from PIL import Image
from vllm import LLM, SamplingParams

# 进行图片推理
MAX_NUM_FRAMES = 16
def encode_video(filepath):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    vr = VideoReader(filepath, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx)>MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    video = vr.get_batch(frame_idx).asnumpy()
    video = [Image.fromarray(v.astype('uint8')) for v in video]
    return video

MODEL_NAME = "checkpoints/MiniCPM-V-2_6" 
# openbmb/MiniCPM-V-2_6: auto download from huggingface
# local path: checkpoints/MiniCPM-V-2_6
llm = LLM(
    model=MODEL_NAME,
    gpu_memory_utilization=0.95,
    max_model_len=4096
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
stop_tokens = ['<|im_end|>', '<|endoftext|>']
stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]


frames = encode_video("your_video.mp4")
messages = [{
    "role":
    "user",
    "content":
    "".join(["(<image>./</image>)"] * len(frames)) + "\nPlease describe this video."
}]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

sampling_params = SamplingParams(
    stop_token_ids=stop_token_ids, 
    #use_beam_search=False
    temperature=0.7,
    top_p=0.8,
    top_k=100, 
    max_tokens=512
)

outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {
        "image": {
            "images": frames,
            "use_image_id": False,
            "max_slice_nums": 1 if len(frames) > 16 else 2
        }
    }
}, sampling_params=sampling_params)
```
- Demo Code For Multiple Images
```python
from transformers import AutoTokenizer
from PIL import Image
from vllm import LLM, SamplingParams

MODEL_NAME = "openbmb/MiniCPM-V-2_6"

image = Image.open("xxx.png").convert("RGB")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    gpu_memory_utilization=1,
    max_model_len=2048
)

messages = [{
    "role":
    "user",
    "content":
    # Number of images
    "(<image>./</image>)" + \
    "\nWhat is the content of this image?" 
}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Single Inference
inputs = {
    "prompt": prompt,
    "multi_modal_data": {
        "image": image
        # Multi images, the number of images should be equal to that of `(<image>./</image>)`
        # "image": [image, image] 
    },
}

# 2.6
stop_tokens = ['<|im_end|>', '<|endoftext|>']
stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

sampling_params = SamplingParams(
    stop_token_ids=stop_token_ids, 
    #use_beam_search=True,
    temperature=0, 
    best_of=1,
    max_tokens=64
)

outputs = llm.generate(inputs, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
```





llama.cpp in CPU mode but use RAM.

./llama-minicpmv-cli -m /home/cxx/HWs/AKS/llama.cpp/Minicpmv2_6_gguf/ggml-model-Q4_K_M.gguf --mmproj /home/cxx/HWs/AKS/llama.cpp/Minicpmv2_6_gguf/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image /home/cxx/HWs/AKS/datasets/img/bto.jpg -p "这张图片中有什么？"

>=8GB
./llama-minicpmv-cli -m /home/cxx/HWs/AKS/llama.cpp/Minicpmv2_6_gguf/ggml-model-Q4_K_M.gguf --mmproj /home/cxx/HWs/AKS/llama.cpp/Minicpmv2_6_gguf/mmproj-model-f16.gguf -c 8192 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --video /home/cxx/HWs/AKS/datasets/XD_violence/ours/Shooting/Test_0_Shooting.mp4 -p "Please answer me question about this video. What is the main action of hero in the video? your candidates: Shooting, Fighting, Running, Walking. You should answer me one or two of candidates. Your answer should be separated by comma and space, Like [Shooting, Fighting]. Without any explanation words."


>=19GB
./llama-minicpmv-cli -m /home/cxx/HWs/AKS/llama.cpp/Minicpmv2_6_gguf/ggml-model-f16.gguf --mmproj /home/cxx/HWs/AKS/llama.cpp/Minicpmv2_6_gguf/mmproj-model-f16.gguf -c 8192 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --video /home/cxx/HWs/AKS/datasets/XD_violence/ours/Shooting/Test_0_Shooting.mp4 -p "Please answer me question about this video. What is the main action of hero in the video? your candidates: Shooting, Fighting, Running, Walking. You should answer me one or two of candidates. Your answer should be separated by comma and space, Like [Shooting, Fighting]. Without any explanation words."