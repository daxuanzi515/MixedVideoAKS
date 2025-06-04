# import json
# keyword = "Fighting/class_1"
# keyword = "Fighting/class_2"
# keyword = "Shooting/class_1"
# keyword = "Shooting/class_2"    



# # File paths
# ori = f"/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/{keyword}/blip/selected_frames/selected_frames.json"
# opt = f"/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/{keyword}/blip/optimized_frames/optimized_frames.json"
# # ex = f"/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_thing/{keyword}/blip/frames.json"
# ex = f"/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/{keyword}/blip/frames.json"

# # Load JSON data
# with open(ori, 'r') as f:
#     data_1 = json.load(f)

# with open(opt, 'r') as f:
#     data_2 = json.load(f)

# with open(ex, 'r') as f:
#     data_3 = json.load(f)

# # Print number of samples
# print(f"Number of samples in ori: {len(data_1)}")
# print(f"Number of samples in opt: {len(data_2)}")
# print(f"Number of samples in ex: {len(data_3)}")

# # Check if all sets have the same number of samples

# def com_sum(data):
#     sum = 0
#     for i in data:
#         num = len(i)
#         sum += num
#     return sum

# print(f"Number of frames in ori: {com_sum(data_1)}")
# print(f"Number of frames in opt: {com_sum(data_2)}")
# print(f"Number of frames in ex: {com_sum(data_3)}")


# # Function to calculate differences between two sets
# def calculate_diff(a, b):
#     set_a, set_b = set(a), set(b)
#     added = sorted(list(set_b - set_a))
#     removed = sorted(list(set_a - set_b))
#     return added, removed

# # Calculate differences between all pairs
# diff_stats_1_2 = []
# diff_stats_1_3 = []
# diff_stats_2_3 = []

# for i, (a, b, c) in enumerate(zip(data_1, data_2, data_3)):
#     added_1_2, removed_1_2 = calculate_diff(a, b)
#     added_1_3, removed_1_3 = calculate_diff(a, c)
#     added_2_3, removed_2_3 = calculate_diff(b, c)

#     if added_1_2 or removed_1_2:
#         diff_stats_1_2.append({
#             "index": i,
#             "added": added_1_2,
#             "removed": removed_1_2
#         })

#     if added_1_3 or removed_1_3:
#         diff_stats_1_3.append({
#             "index": i,
#             "added": added_1_3,
#             "removed": removed_1_3
#         })

#     if added_2_3 or removed_2_3:
#         diff_stats_2_3.append({
#             "index": i,
#             "added": added_2_3,
#             "removed": removed_2_3
#         })

# # Print differences
# print(f"\nTotal different entries between ori and opt: {len(diff_stats_1_2)}")
# print(f"Total different entries between ori and ex: {len(diff_stats_1_3)}")
# print(f"Total different entries between opt and ex: {len(diff_stats_2_3)}")

# # Print some examples of differences
# print("\nExamples of differences between ori and opt:")
# for d in diff_stats_1_2[:5]:  # Print first 5 differences
#     print(f"Sample #{d['index']}")
#     print(f"  ➕ Added in opt: {d['added']}")
#     print(f"  ➖ Removed from ori: {d['removed']}")

# print("\nExamples of differences between ori and ex:")
# for d in diff_stats_1_3[:5]:  # Print first 5 differences
#     print(f"Sample #{d['index']}")
#     print(f"  ➕ Added in ex: {d['added']}")
#     print(f"  ➖ Removed from ori: {d['removed']}")

# print("\nExamples of differences between opt and ex:")
# for d in diff_stats_2_3[:5]:  # Print first 5 differences
#     print(f"Sample #{d['index']}")
#     print(f"  ➕ Added in ex: {d['added']}")
#     print(f"  ➖ Removed from opt: {d['removed']}")

# # Function to calculate Jaccard similarity
# def jaccard(a, b):
#     a, b = set(a), set(b)
#     if not a and not b:
#         return 1.0
#     return len(a & b) / len(a | b)

# # Calculate Jaccard similarity for all pairs
# scores_1_2 = [jaccard(a, b) for a, b in zip(data_1, data_2)]
# scores_1_3 = [jaccard(a, c) for a, c in zip(data_1, data_3)]
# scores_2_3 = [jaccard(b, c) for b, c in zip(data_2, data_3)]

# print(f"\nAverage frame overlap (Jaccard) between ori and opt: {sum(scores_1_2)/len(scores_1_2):.4f}")
# print(f"Average frame overlap (Jaccard) between ori and ex: {sum(scores_1_3)/len(scores_1_3):.4f}")
# print(f"Average frame overlap (Jaccard) between opt and ex: {sum(scores_2_3)/len(scores_2_3):.4f}")


import json

# List of keywords for the four classes
keywords = [
    "Fighting/class_1",
    "Fighting/class_2",
    "Shooting/class_1",
    "Shooting/class_2"
]

# Initialize lists to store metrics
num_samples_ori = []
num_samples_opt = []
num_samples_ex = []
num_frames_ori = []
num_frames_opt = []
num_frames_ex = []
total_diff_1_2 = []
total_diff_1_3 = []
total_diff_2_3 = []
jaccard_scores_1_2 = []
jaccard_scores_1_3 = []
jaccard_scores_2_3 = []

# Function to calculate differences between two sets
def calculate_diff(a, b):
    set_a, set_b = set(a), set(b)
    added = sorted(list(set_b - set_a))
    removed = sorted(list(set_a - set_b))
    return added, removed

# Function to calculate Jaccard similarity
def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

# Loop through each class
for keyword in keywords:
    # File paths
    ori = f"/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/{keyword}/blip/selected_frames/selected_frames.json"
    opt = f"/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/{keyword}/blip/optimized_frames/optimized_frames.json"
    ex = f"/home/cxx/HWs/AKS/Mixed Video Pipeline/extracted_things/{keyword}/blip/frames.json"

    # Load JSON data
    with open(ori, 'r') as f:
        data_1 = json.load(f)

    with open(opt, 'r') as f:
        data_2 = json.load(f)

    with open(ex, 'r') as f:
        data_3 = json.load(f)

    # Store number of samples
    num_samples_ori.append(len(data_1))
    num_samples_opt.append(len(data_2))
    num_samples_ex.append(len(data_3))

    # Function to compute total number of frames
    def com_sum(data):
        return sum(len(i) for i in data)

    # Store number of frames
    num_frames_ori.append(com_sum(data_1))
    num_frames_opt.append(com_sum(data_2))
    num_frames_ex.append(com_sum(data_3))

    # Calculate differences between all pairs
    diff_stats_1_2 = []
    diff_stats_1_3 = []
    diff_stats_2_3 = []

    for i, (a, b, c) in enumerate(zip(data_1, data_2, data_3)):
        added_1_2, removed_1_2 = calculate_diff(a, b)
        added_1_3, removed_1_3 = calculate_diff(a, c)
        added_2_3, removed_2_3 = calculate_diff(b, c)

        if added_1_2 or removed_1_2:
            diff_stats_1_2.append({
                "index": i,
                "added": added_1_2,
                "removed": removed_1_2
            })

        if added_1_3 or removed_1_3:
            diff_stats_1_3.append({
                "index": i,
                "added": added_1_3,
                "removed": removed_1_3
            })

        if added_2_3 or removed_2_3:
            diff_stats_2_3.append({
                "index": i,
                "added": added_2_3,
                "removed": removed_2_3
            })

    # Store total differences
    total_diff_1_2.append(len(diff_stats_1_2))
    total_diff_1_3.append(len(diff_stats_1_3))
    total_diff_2_3.append(len(diff_stats_2_3))

    # Calculate Jaccard similarity for all pairs
    scores_1_2 = [jaccard(a, b) for a, b in zip(data_1, data_2)]
    scores_1_3 = [jaccard(a, c) for a, c in zip(data_1, data_3)]
    scores_2_3 = [jaccard(b, c) for b, c in zip(data_2, data_3)]

    # Store Jaccard scores
    jaccard_scores_1_2.append(sum(scores_1_2) / len(scores_1_2))
    jaccard_scores_1_3.append(sum(scores_1_3) / len(scores_1_3))
    jaccard_scores_2_3.append(sum(scores_2_3) / len(scores_2_3))

# Compute averages
avg_num_samples_ori = sum(num_samples_ori) / len(num_samples_ori)
avg_num_samples_opt = sum(num_samples_opt) / len(num_samples_opt)
avg_num_samples_ex = sum(num_samples_ex) / len(num_samples_ex)
avg_num_frames_ori = sum(num_frames_ori) / len(num_frames_ori)
avg_num_frames_opt = sum(num_frames_opt) / len(num_frames_opt)
avg_num_frames_ex = sum(num_frames_ex) / len(num_frames_ex)
avg_total_diff_1_2 = sum(total_diff_1_2) / len(total_diff_1_2)
avg_total_diff_1_3 = sum(total_diff_1_3) / len(total_diff_1_3)
avg_total_diff_2_3 = sum(total_diff_2_3) / len(total_diff_2_3)
avg_jaccard_scores_1_2 = sum(jaccard_scores_1_2) / len(jaccard_scores_1_2)
avg_jaccard_scores_1_3 = sum(jaccard_scores_1_3) / len(jaccard_scores_1_3)
avg_jaccard_scores_2_3 = sum(jaccard_scores_2_3) / len(jaccard_scores_2_3)

# Print averages
print(f"\nAverage number of samples in ori: {avg_num_samples_ori:.2f}")
print(f"Average number of samples in opt: {avg_num_samples_opt:.2f}")
print(f"Average number of samples in ex: {avg_num_samples_ex:.2f}")
print(f"Average number of frames in ori: {avg_num_frames_ori:.2f}")
print(f"Average number of frames in opt: {avg_num_frames_opt:.2f}")
print(f"Average number of frames in ex: {avg_num_frames_ex:.2f}")
print(f"Average total different entries between ori and opt: {avg_total_diff_1_2:.2f}")
print(f"Average total different entries between ori and ex: {avg_total_diff_1_3:.2f}")
print(f"Average total different entries between opt and ex: {avg_total_diff_2_3:.2f}")
print(f"Average frame overlap (Jaccard) between ori and opt: {avg_jaccard_scores_1_2:.4f}")
print(f"Average frame overlap (Jaccard) between ori and ex: {avg_jaccard_scores_1_3:.4f}")
print(f"Average frame overlap (Jaccard) between opt and ex: {avg_jaccard_scores_2_3:.4f}")