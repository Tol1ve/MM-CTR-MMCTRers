# 🚀 MMCTR Challenge - Team MMCTR from xinlab 🔥

## 🏆 Introduction

This repository contains our solution for the **WWW 2025 Multimodal CTR Prediction Challenge** (Track 2: Multimodal CTR Prediction), organized by the **WWW 2025 EReL@MIR workshop**. The challenge focuses on developing effective multimodal representation learning and fusion methods for CTR prediction in recommendation systems.

Our solution is based on the baseline provided by the challenge organizers, with modifications and improvements in feature fusion and model tuning. We aim to enhance the recommendation accuracy by optimizing multimodal embeddings and their integration into the CTR model.

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
cd ~/MMCTR_Challenge/data/
find -L .
```

📁 **Expected data structure:**
```
./MicroLens_1M_x1/
├── train.parquet
├── valid.parquet
├── test.parquet
├── item_info.parquet
├── item_feature.parquet   
├── item_emb.parquet      
├── item_seq.parquet      
├── item_images.rar      
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

```bash
bash run.sh
```

 **The `run.sh` script will handle:**
1️⃣ Data preprocessing 🏗️
2️⃣ Model training 🎯
3️⃣ Prediction on the test set 🔍
4️⃣ Generating the final submission file 📄

---

## 💬 4. Discussion

**For any questions or discussions related to our implementation, feel free to contact us via email.**

 **Contact:** [scut201930033162@gmail.com]

 Alternatively, you can start a discussion on the challenge forum:
- 🔗 [Codabench Forum](https://www.codabench.org/forums/5287/)



---

🔥 This document serves as the official README for our MMCTR challenge submission. Thank you! 🚀

