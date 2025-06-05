# RECORD

#### Total Table:

| Method            | Task Type         | General Correct Rate (%) | True Positive Rate (%) | False Positive Rate (%) | True Negative Rate (%) |
|-------------------|-------------------|--------------------------|------------------------|-------------------------|------------------------|
| **MixedVideoAKS** | Fighting_Class_1  | **94.25** (86.21)        | **43.68** (40.23)      | **50.57** (45.98)       | **6.90** (13.79)       |
| **MixedVideoAKS** | Fighting_Class_2  | **93.33** (85.56)        | **27.78** (24.44)      | **65.56** (66.67)       | **6.67** (8.89)       |
| **MixedVideoAKS** | Shooting_Class_1  | **97.78** (88.89)        | **46.67** (44.44)      | **51.11** (44.44)       | **2.22** (5.56)        |
| **MixedVideoAKS** | Shooting_Class_2  | **97.75** (91.01)        | **49.44** (46.07)      | **48.31** (44.94)       | **2.25** (3.37)        |

### Our Method: MixedVideoAKS

#### Fighting/class_1/jpgs
- **Sampling**: From keyframes group 4, the 3rd one jpg.
- **Total Items**: 87
For example, in `/home/cxx/HWs/AKS/Mixed Video Pipeline/Testing/Fighting_class_1_res.txt`:
```
['Fighting', 'Fighting', 'Fighting', 'Fighting', 'Shooting', 'Fighting', 'Fighting', 'NONE', 'NONE', 'NONE', 'Fighting', 'NONE', 'Fighting', 'Fighting', 'NONE', 'Fighting', 'Shooting', 'Fighting', 'Fighting', 'Fighting', 'Fighting', 'NONE', 'NONE', 'NONE', 'Fighting', 'Shooting', 'Fighting', 'NONE', 'NONE', 'Fighting', 'NONE', 'Fighting', 'NONE', 'NONE', 'Fighting', 'NONE', 'NONE', 'Shooting', 'NONE', 'Fighting', 'NONE', 'NONE', 'NONE', 'Fighting', 'NONE', 'Fighting', 'Shooting', 'NONE', 'UNKNOWN', 'Fighting', 'Fighting', 'NONE', 'NONE', 'Fighting', 'NONE', 'Fighting', 'NONE', 'NONE', 'NONE', 'NONE', 'Fighting', 'NONE', 'Fighting', 'Fighting', 'Fighting', 'Fighting', 'NONE', 'Fighting', 'Fighting', 'NONE', 'NONE', 'NONE', 'NONE', 'Fighting', 'NONE', 'Fighting', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'Fighting', 'Fighting', 'Fighting', 'NONE', 'NONE']
```

- **Shooting number**: 5, wrong answer
- **Fighting number**: 38, correct answer
- **NONE number**: 44, correct answer
- **UNKNOWN number**: 1, wrong answer

- **General Correct Rate**: $\frac{38 + 44}{87} \times 100\% = 94.25\%$
- **True Positive Rate**: $\frac{38}{87} \times 100\% = 43.68\%$
- **False Positive Rate**: $\frac{44}{87} \times 100\% = 50.57\%$
- **True Negative Rate**: $\frac{1 + 5}{87} \times 100\% = 6.90\%$

#### Fighting/class_2/jpgs
- **Sampling**: From keyframes group 4, the 3rd one jpg.
- **Total Items**: 90
- **Shooting number**: 6, wrong answer
- **Fighting number**: 25, correct answer
- **NONE number**: 59, correct answer
- **UNKNOWN number**: 0, wrong answer

- **General Correct Rate**: $\frac{25 + 59}{90} \times 100\% = 93.33\%$
- **True Positive Rate**: $\frac{25}{90} \times 100\% = 27.78\%$
- **False Positive Rate**: $\frac{59}{90} \times 100\% = 65.56\%$
- **True Negative Rate**: $\frac{6}{90} \times 100\% = 6.67\%$


#### Shooting/class_1/jpgs
- **Sampling**: From keyframes group 4, the 3rd one jpg.
- **Total Items**: 90
- **Fighting number**: 2, wrong answer
- **Shooting number**: 42, correct answer
- **NONE number**: 46, correct answer
- **UNKNOWN number**: 0, wrong answer

- **General Correct Rate**: $\frac{42 + 46}{90} \times 100\% = 97.78\%$
- **True Positive Rate**: $\frac{42}{90} \times 100\% = 46.67\%$
- **False Positive Rate**: $\frac{46}{90} \times 100\% = 51.11\%$
- **True Negative Rate**: $\frac{2}{90} \times 100\% = 2.22\%$

#### Shooting/class_2/jpgs
- **Sampling**: From keyframes group 4, the 3rd one jpg.
- **Total Items**: 89
- **Fighting number**: 2, wrong answer
- **Shooting number**: 44, correct answer
- **NONE number**: 43, correct answer
- **UNKNOWN number**: 0, wrong answer

- **General Correct Rate**: $\frac{44 + 43}{89} \times 100\% = 97.75\%$
- **True Positive Rate**: $\frac{44}{89} \times 100\% = 49.44\%$
- **False Positive Rate**: $\frac{43}{89} \times 100\% = 48.31\%$
- **True Negative Rate**: $\frac{2}{89} \times 100\% = 2.25\%$


## Baseline: AKS

- Fighting/class_1/jpgs, sampling from keyframes group 4 the 3rd one jpg, in total 87 items.
  - Fighting (Correct): 35
  - Shooting (Wrong): 7
  - NONE (Correct): 40
  - UNKNOWN (Wrong): 5
  - **General Correct Rate**: $\frac{35 + 40}{87} \times 100\% \approx 86.21\%$
  - **True Positive Rate**: $\frac{35}{87} \times 100\% \approx 40.23\%$
  - **False Positive Rate**: $\frac{40}{87} \times 100\% \approx 45.98\%$
  - **True Negative Rate**: $\frac{5 + 7}{87} \times 100\% \approx 13.79\%$


- Fighting/class_2/jpgs, sampling from keyframes group 4 the 3rd one jpg, in total 90 items.
  - Fighting (Correct): 22
  - Shooting (Wrong): 8
  - NONE (Correct): 60
  - UNKNOWN (Wrong): 0
  - **General Correct Rate**: $\frac{22 + 55}{90} \times 100\% \approx 85.56\%$
  - **True Positive Rate**: $\frac{22}{90} \times 100\% \approx 24.44\%$
  - **False Positive Rate**: $\frac{60}{90} \times 100\% \approx 66.67\%$
  - **True Negative Rate**: $\frac{8}{90} \times 100\% \approx 8.89\%$

- Shooting/class_1/jpgs, sampling from keyframes group 4 the 3rd one jpg, in total 90 items.
  - Fighting (Wrong): 5
  - Shooting (Correct): 40
  - NONE (Correct): 40
  - UNKNOWN (Wrong): 5
  - **General Correct Rate**: $\frac{40 + 40}{90} \times 100\% \approx 88.89\%$
  - **True Positive Rate**: $\frac{40}{90} \times 100\% \approx 44.44\%$
  - **False Positive Rate**: $\frac{40}{90} \times 100\% \approx 44.44\%$
  - **True Negative Rate**: $\frac{5}{90} \times 100\% \approx 5.56\%$

- Shooting/class_2/jpgs, sampling from keyframes group 4 the 3rd one jpg, in total 89 items.
  - Fighting (Wrong): 5
  - Shooting (Correct): 41
  - NONE (Correct): 40
  - UNKNOWN (Wrong): 3
  - **General Correct Rate**: $\frac{41 + 40}{89} \times 100\% \approx 91.01\%$
  - **True Positive Rate**: $\frac{41}{89} \times 100\% \approx 46.07\%$
  - **False Positive Rate**: $\frac{40}{89} \times 100\% \approx 44.94\%$
  - **True Negative Rate**: $\frac{3}{89} \times 100\% \approx 3.37\%$
