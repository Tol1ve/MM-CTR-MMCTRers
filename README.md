# 🚀 MMCTR Challenge - Team MMCTRers from xinlab 

## 🏆 Introduction

This repository contains our solution for the **WWW 2025 Multimodal CTR Prediction Challenge** (Track 2: Multimodal CTR Prediction), organized by the **WWW 2025 EReL@MIR workshop**. The challenge focuses on developing effective multimodal representation learning and fusion methods for CTR prediction in recommendation systems.

Our solution is based on the baseline provided by the challenge organizers, with modifications and improvements in model tuning. We aim to enhance the recommendation accuracy by optimizing a new feature embedding mechanism and an  auxiliary loss.

📌 For details about the challenge, please visit:
-  [Challenge website](https://erel-mir.github.io/challenge/mmctr-track2/)
-  [Competition platform](https://www.codabench.org/competitions/5372/)
-  [Baseline repository](https://github.com/reczoo/WWW2025_MMCTR_Challenge)

---

## 📂 1. Data Preparation

 **Download the dataset** from the official source:
   - [🔗 MicroLens 1M MMCTR Dataset](https://recsys.westlake.edu.cn/MicroLens_1M_MMCTR)

 **Unzip the dataset** into the `data/` directory:

```bash
cd ~/MM-CTR-MMCTRers/data/
find -L .
```

📁 **Expected data structure:**
```
./MicroLens_1M_x1/
├── train.parquet
├── valid.parquet
├── test.parquet
├── item_info.parquet
item_feature.parquet   
item_emb.parquet      
item_seq.parquet      
item_images.rar      
```

---

## 🛠️ 2. Environment Setup

We use the same environment setup as the baseline:

```bash
conda create -n fuxictr python==3.9
pip install -r requirements.txt
source activate fuxictr
```

✅ **Required dependencies:**
- 🟢 torch==1.13.1+cu117
- 🟢 fuxictr==2.3.7

📌 **For full setup details**, please refer to the [baseline repository](https://github.com/reczoo/WWW2025_MMCTR_Challenge).

---

## 🚀 3. How to Run

After setting up the environment and downloading the dataset, simply run:

✅ **suggested**
```bash
chmod +x run.sh
bash run.sh
```

⚠️ **unproved way**
```bash
chmod +x run_from_begin.sh
bash run_from_begin.sh
```

🚀 **inferrence from our pre-trained model,simply:**
```bash
python prediction.py --config ./config/DIN_attn_emb_v3 --expid DIN_DIN_attn_emb_v2_001_acf37100 --gpu 0
```
 **The `run.sh` script will handle:**

1️⃣ config set(if not ) 🏗️
2️⃣ Model training 🎯

3️⃣ Prediction on the test set 🔍
4️⃣ Generating the final submission file 📄

Our pre-trained model is located at 

  - ./checkpoints/DIN_attn_emb_v2/DIN_DIN_attn_emb_v2_001_acf37100.model

**You can also modify our model and parameters as you want it by changing  ./config/DIN_attn_emb_v3**

⚠️ **Notice, we upload some different version of models we have tried. the final model we use for the competition is at ./src/DIN_att_emb_rms.py**
---

## 💬 4. Discussion

**For any questions or discussions related to our implementation, feel free to contact us via email.**

 **Contact:** [scut201930033162@gmail.com]

---
## 📌 TODO：
    - detailed illustration of proposed algorithm
    - A readme in chinese
---

This document serves as the official README for our MMCTR challenge submission. Thank you! 

🔥 If you find this work helpful , Please cite the officer paper:

+ Jieming Zhu, Jinyang Liu, Shuai Yang, Qi Zhang, Xiuqiang He. [Open Benchmarking for Click-Through Rate Prediction](https://arxiv.org/abs/2009.05794). *The 30th ACM International Conference on Information and Knowledge Management (CIKM)*, 2021.
