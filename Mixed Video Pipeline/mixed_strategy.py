from moviepy import VideoFileClip, concatenate_videoclips
import os


class VideoMixer:
    def __init__(self, benign_video, malignant_video, output_path):
        self.benign_video = benign_video
        self.malignant_video = malignant_video
        self.output_path = output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def _load_clip(self, video_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        return VideoFileClip(video_path)

    def _resize_to_match(self, clip, target_size):
        """Resize a clip to match the target size (width, height)."""
        return clip.resized(target_size)

    def _loop_fill_clip(self, clip, target_duration):
        """
        Ensure a clip reaches the target_duration by looping and concatenating itself.
        """
        if clip.duration >= target_duration:
            return clip.subclipped(0, target_duration)
        
        clips = []
        remaining = target_duration
        while remaining > 0:
            sub = clip.subclipped(0, min(clip.duration, remaining))
            clips.append(sub)
            remaining -= sub.duration
        return concatenate_videoclips(clips)

    def generate_mixed_video(self, class_type, total_duration=60, customs=None):
        """
        Generate mixed video based on class_type (1 or 2), or a custom ratio (class_type=3).
        """
        if class_type == 1:
            B1_ratio, M_ratio, B2_ratio = 0.5, 0.3, 0.1
        elif class_type == 2:
            B1_ratio, M_ratio, B2_ratio = 0.75, 0.2, 0.05
        elif customs is not None and class_type == 3:
            B1_ratio, M_ratio, B2_ratio = customs
        else:
            raise ValueError("Unsupported class_type. Use 1, 2, or 3 with custom ratios.")

        B1_dur = int(total_duration * B1_ratio)
        M_dur = int(total_duration * M_ratio)
        B2_dur = int(total_duration * B2_ratio)

        benign_clip = self._load_clip(self.benign_video)
        malignant_clip = self._load_clip(self.malignant_video)

        target_size = benign_clip.size

        b1 = self._resize_to_match(self._loop_fill_clip(benign_clip, B1_dur), target_size)
        m  = self._resize_to_match(self._loop_fill_clip(malignant_clip, M_dur), target_size)
        b2 = self._resize_to_match(self._loop_fill_clip(benign_clip, B2_dur), target_size)

        final_clip = concatenate_videoclips([b1, m, b2])
        final_path = self.output_path.replace(".mp4", f"_class{class_type}_{total_duration}s.mp4")
        final_clip.write_videofile(final_path, codec="libx264")

        print(f"[✅] Mixed video saved to {final_path}")


if __name__ == "__main__":
    Number = 30
    class_type = 1
    total_duration = 360
    B_folder = "datasets/XD_violence/ours/Benign/"

    B_list = sorted([
        os.path.join(B_folder, f)
        for f in os.listdir(B_folder)
        if f.endswith(".mp4")
    ])[:Number]


    M_list = [f"datasets/XD_violence/ours/Fighting/Test_{i}_Fighting.mp4" for i in range(Number)]
    O_list = [f"Mixed Video Pipeline/outputs/class_1/Test_{i}_Mixed.mp4" for i in range(Number)]

    for i in range(0, 1):
        print(f"[🚀] Processing sample Class {class_type}: {i}...")
        print(f"[🤖] Benign video: {B_list[i]}, Malignant video: {M_list[i]}")
        mixer = VideoMixer(
            benign_video=B_list[i],
            malignant_video=M_list[i],
            output_path=O_list[i]
        )
        mixer.generate_mixed_video(class_type=class_type, total_duration=total_duration)
