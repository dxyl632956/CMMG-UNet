# CMMG-UNet
**Semantic-Driven Lesion Segmentation in Chest Images with Clustered Memory Bank and Multi-Granularity Fusion**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

This is the official PyTorch implementation of **CMMG-UNet**. 

> **Abstract:** Multi-modal learning has emerged as a promising direction for medical image segmentation, yet existing methods grapple with "mode collapse" in memory mechanisms and prohibitive computational costs. In this paper, we propose CMMG-UNet, a novel asymmetric framework designed for efficient and robust lesion segmentation. We introduce a **Clustered and Loss-Gated Memory Bank (CLMB)** to retain "hard samples" and mitigate catastrophic forgetting. To bridge the semantic gap, a **Multi-Granularity Textual-Visual Fusion (MTVF)** module is devised to hierarchically align visual features with text semantics. CMMG-UNet achieves state-of-the-art performance with a highly efficient computational footprint of only 21.0G FLOPs.

## 🚀 Framework
![](D:\CMMG-UNet\struct.png)

## 📊 Main Results
Performance comparison on QaTa-COV19 and MosMedData+ datasets:

| Dataset         | Param (M) | FLOPs (G) |  DSC (%)  |  IoU (%)  |
| :-------------- | :-------: | :-------: | :-------: | :-------: |
| **QaTa-COV19**  |   58.9    | **21.0**  | **91.45** | **84.22** |
| **MosMedData+** |   58.9    | **21.0**  | **79.07** | **65.41** |

## 💻 Requirements
1. **Environment**: The main mandatory dependency versions are as follows:  
    ```text
    python=3.10.15  
    torch=2.5.0+cu12.1  
    torchvision=0.13.1  
    pytorch_lightning=1.9.0  
    torchmetrics=0.10.3  
    transformers=4.24.0  
    monai=1.0.1  
    pandas  
    einops  
    ```

2. **(Optional) Download the pretrained models**:
   
   * **CXR-BERT-specialized**: [Huggingface Link](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized/tree/main)  
   * **ConvNeXt-tiny**: [Huggingface Link](https://huggingface.co/facebook/convnext-tiny-224/tree/main)

   Download the `pytorch_model.bin` file into `./lib/BiomedVLP-CXR-BERT-specialized/` and `./lib/convnext-tiny-224/`.

   *Alternatively, you can load these models directly online:*
   
   ```python
   url = "microsoft/BiomedVLP-CXR-BERT-specialized"
   tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
   model = AutoModel.from_pretrained(url, trust_remote_code=True)
   ```

## 🗂 Dataset

1. **QaTa-COV19 Dataset** (Images & Segmentation Masks)

   [QaTa-COV19 Dataset on Kaggle](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset)

   *Note: We use QaTa-COV19-v2 in our experiments.*

2. **MosMedData+ Dataset** (Images & Segmentation Masks)

   [MosMedData+ Dataset on Kaggle](https://www.kaggle.com/datasets/mathurinache/mosmeddata-chest-ct-scans-with-covid19/data)

3. **QaTa-COV19 And MosMedData+ Text Annotations** (from third-party)

   Check out the related annotations provided by [LViT-TMI'23](https://github.com/HUANGLIZI/LViT).

   *Thanks to Li et al. for their contributions.*

## 🏃 Quick Start

Our training pipeline is implemented based on PyTorch Lightning. Please check the relevant training settings in `train.py` and `config`.

*Example path setting:*

Plaintext

```
train_csv_path: ./data/QaTa-COV19-v2/prompt/train.csv
```

**To train the model:**

Bash

```
python train.py
```

**To evaluate the model:**

Bash

```
python evaluate.py
```

Our checkpoints will be made public after the paper is accepted.



## 🙏 Acknowledgements

This codebase is built upon [MMI-UNet](https://www.google.com/search?q=https://github.com/PhuocNguyenBui/MMI-UNet) and [GuideDecoder](https://github.com/Junelin2333/LanGuideMedSeg-MICCAI2023). We sincerely thank the authors for making their excellent source code publicly available.