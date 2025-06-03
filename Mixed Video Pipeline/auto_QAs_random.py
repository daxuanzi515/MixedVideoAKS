# For each video, generate 4 different QA pairs with varied questions and options.

import random

# Expanded question templates for each type
qa_templates_expanded = {
    "Fighting": [
        {
            "question": "What violent behavior involving physical confrontation is shown in this video?",
            "choices": ["Fighting", "Shooting", "Running", "Standing"],
            "answer": "Fighting"
        },
        {
            "question": "Which type of aggressive hand-to-hand action does this video illustrate?",
            "choices": ["Shooting", "Fighting", "Dancing", "Sitting"],
            "answer": "Fighting"
        },
        {
            "question": "What kind of direct physical violence is depicted in this scene?",
            "choices": ["Shooting", "Hugging", "Fighting", "Waving"],
            "answer": "Fighting"
        },
        {
            "question": "This video shows a person engaging in what kind of violent activity?",
            "choices": ["Shooting", "Sleeping", "Fighting", "Walking"],
            "answer": "Fighting"
        }
    ],
    "Shooting": [
        {
            "question": "What kind of violent act involving firearms is taking place in this video?",
            "choices": ["Shooting", "Fighting", "Jumping", "Walking"],
            "answer": "Shooting"
        },
        {
            "question": "What type of weapon-based violence is happening in this scene?",
            "choices": ["Fighting", "Shooting", "Driving", "Talking"],
            "answer": "Shooting"
        },
        {
            "question": "Which violent action involving a gun is depicted in the video?",
            "choices": ["Fighting", "Cooking", "Shooting", "Reading"],
            "answer": "Shooting"
        },
        {
            "question": "What behavior involving the use of a firearm is shown here?",
            "choices": ["Shooting", "Sleeping", "Fighting", "Eating"],
            "answer": "Shooting"
        }
    ]
}

# Generate dataset with 4 QAs per video

behaviors = ["Fighting", "Shooting"]
classes = ["class_1", "class_2"]

qa_records_shuffled = []

for behavior in behaviors:
    for class_type in classes:
        for i in range(30):
            filename = f"Test_{i}_Mixed.mp4"
            path = f"/home/cxx/HWs/AKS/Mixed Video Pipeline/outputs/{behavior}/{class_type}/{filename}"
            questions_pool = qa_templates_expanded[behavior][:]
            random.shuffle(questions_pool)
            sampled_qas = questions_pool[:4]  # select 4 randomly per video
            for qa in sampled_qas:
                qa_records_shuffled.append({
                    "video_id": f"{behavior}/{class_type}/Test_{i}_Mixed.mp4",
                    "video_path": path,
                    "question": qa["question"],
                    "choices": qa["choices"],
                    "answer": qa["answer"]
                })

# Save corrected version
import json
output_multi_path = "Mixed Video Pipeline/outputs/mixed_video_qa_randomized_4x_per_video.json"
with open(output_multi_path, "w") as f:
    json.dump(qa_records_shuffled, f, indent=2)

print(f"Saved {len(qa_records_shuffled)} QA records to {output_multi_path}")