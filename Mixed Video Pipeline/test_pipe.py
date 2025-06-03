from mixed_strategy import VideoMixer
import os

if __name__ == "__main__":
    Number = 30
    class_type = 2
    total_duration = 360
    B_folder = "/home/cxx/HWs/AKS/datasets/XD_violence/ours/Benign/"

    B_list = sorted([
        os.path.join(B_folder, f)
        for f in os.listdir(B_folder)
        if f.endswith(".mp4")
    ])[:Number]


    M_list = [f"/home/cxx/HWs/AKS/datasets/XD_violence/ours/Fighting/Test_{i}_Fighting.mp4" for i in range(Number)]
    O_list = [f"/home/cxx/HWs/AKS/Mixed Video Pipeline/outputs/class_2/Test_{i}_Mixed.mp4" for i in range(Number)]

    for i in range(30):
        print(f"[🚀] Processing sample Class {class_type}: {i}...")
        print(f"[🤖] Benign video: {B_list[i]}, Malignant video: {M_list[i]}")
        mixer = VideoMixer(
            benign_video=B_list[i],
            malignant_video=M_list[i],
            output_path=O_list[i]
        )
        mixer.generate_mixed_video(class_type=class_type, total_duration=total_duration)


