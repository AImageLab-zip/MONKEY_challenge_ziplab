## UNIMORE AImageLab Zip MONKEY Challenge Solution

## Overview of the MONKEY Challenge: Detection of Inflammation in Kidney Biopsies

The MONKEY (Machine-learning for Optimal detection of iNflammatory cells in the KidnEY) challenge aims to develop automated methods for detecting and classifying inflammatory cells in kidney transplant biopsies. This initiative seeks to enhance the consistency and efficiency of histopathological assessments, particularly in the context of the Banff classification system.

### Challenge Overview

The challenge comprises two primary tasks:

1. **Detection of Mononuclear Inflammatory Cells (MNLs):** Identifying mononuclear leukocytes in biopsy images.
2. **Classification of Inflammatory Cells:** Distinguishing between monocytes and lymphocytes within the detected cells.

Participants are provided with baseline models and tutorials to facilitate data handling and model training. Submissions should include coordinate text files for MNLs, monocytes, and lymphocytes. Top-performing teams will receive monetary awards and may be invited to co-author resulting publications.

### Timeline

The challenge is structured into three phases:

1. **Debugging Phase (August–December 2024):** Participants develop and debug their Docker containers with access to debugging logs.
2. **Live Leaderboard Phase (August–November 2024):** Teams submit their models for evaluation on a validation set, with results displayed on a live leaderboard.
3. **Final Test Phase (December 2024):** Final submissions are evaluated on a test set to determine the winners.

An open submission cycle will follow, allowing ongoing participation and submissions for up to five years.

### Dataset

The dataset includes 153 whole-slide images (WSIs) from six pathology departments, with 231 regions of interest annotated for monocytes and lymphocytes. Each case provides:

- 1–3 PAS-stained slide scans.
- One immunohistochemistry (IHC) slide scan (CD3/CD20 and PU.1 staining).

All slides are co-registered, with IHC staining guiding annotations on the PAS slides.

### Submission

Participants must submit Docker containers encapsulating their algorithms. Submissions are evaluated using Free Response Operating Characteristic (FROC) analysis, focusing on sensitivity at predefined false positive rates.

### Evaluation & Ranking

The primary evaluation metric is the FROC score, calculated by assessing sensitivity at specific false positive rates per square millimeter. Separate FROC scores are computed for:

- Overall inflammation cell detection (MNLs).
- Individual detection of monocytes and lymphocytes.

Detailed evaluation scripts are available on the challenge's GitHub repository.

### Rules

Key participation rules include:

- Forming teams (even if consisting of a single participant).
- Each participant can only be a member of one team.
- Anonymous participation is not allowed.
- Submissions must be fully automated methods in Docker containers.
- External data and pre-trained models are permitted if publicly available under a permissive open-source license.

Top-ranking teams may be invited to co-author publications resulting from the challenge.

### Pathology Background

The challenge focuses on detecting mononuclear cells—specifically monocytes and lymphocytes—in kidney transplant biopsies. These cells are crucial in assessing transplant rejection. While morphological differences exist between these cell types, immunohistochemistry (IHC) staining enhances differentiation, with CD3/CD20 marking lymphocytes and PU.1 marking monocytes.


### Organizers

The challenge is organized by a team of professionals specializing in computational pathology and machine learning. For inquiries, participants can contact the organizers through the provided email addresses on the challenge website.

For more detailed information, visit the [MONKEY challenge website](https://monkey.grand-challenge.org/). 