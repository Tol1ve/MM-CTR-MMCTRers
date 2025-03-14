## WWW2025_MMCTR_Challenge

The WWW 2025 Multimodal CTR Prediction Challenge: https://www.codabench.org/competitions/5372/

The MM-CTR challenge is organized by the WWW 2025 EReL@MIR workshop, which contains two sub-tasks: multimodal item embedding and multimodal CTR prediction. The first task centers on developing multimodal representation learning and fusion methods tailored for recommendation tasks, while the second focuses on designing CTR prediction models that effectively utilize embedding features to enhance recommendation accuracy. The two challenge tasks are designed to promote potential solutions with practical value and insights for industrial applications. Please check out more details on the challenge website: https://erel-mir.github.io/challenge/mmctr-track2/.

This baseline is built on top of [FuxiCTR, a configurable, tunable, and reproducible library for CTR prediction](https://github.com/reczoo/FuxiCTR). The library has been listed among [the recommended frameworks](https://github.com/ACMRecSys/recsys-evaluation-frameworks) by the ACM RecSys Conference. We open source the baseline solution code to help beginers get familar with FuxiCTR and quickly get started on this task.

🔥 Please cite the paper:

+ Jieming Zhu, Jinyang Liu, Shuai Yang, Qi Zhang, Xiuqiang He. [Open Benchmarking for Click-Through Rate Prediction](https://arxiv.org/abs/2009.05794). *The 30th ACM International Conference on Information and Knowledge Management (CIKM)*, 2021.


### Data Preparation

1. Download the datasets at: https://recsys.westlake.edu.cn/MicroLens_1M_MMCTR

2. Unzip the data files to the `data` directory

    ```bash
    cd ~/WWW2025_MMCTR_Challenge/data/
    find -L .

    .
    ./MicroLens_1M_x1
    ./MicroLens_1M_x1/train.parquet
    ./MicroLens_1M_x1/valid.parquet
    ./MicroLens_1M_x1/test.parquet
    ./MicroLens_1M_x1/item_info.parquet
    ./item_feature.parquet
    ./item_emb.parquet   
    ./item_seq.parquet  
    ./item_images.rar  
    ```

### Environment

We run the experiments on a P100 GPU server with 16G GPU memory and 750G RAM.

Please set up the environment as follows. 

+ torch==1.13.1+cu117
+ fuxictr==2.3.7

```
conda create -n fuxictr python==3.9
pip install -r requirements.txt
source activate fuxictr
```

### How to Run

1. Train the model on train and validation sets:

    ```
    python run_param_tuner.py --config config/DIN_microlens_mmctr_tuner_config_01.yaml --gpu 0
    ```

    In this config file, you can tune the hyper-parameters accordingly by specifying hyper-parameters as a list for grid search as follows. You could also modify the hyper-parameters directly, e.g., `net_dropout: 0.2`.

    ```
    embedding_regularizer: [1.e-6, 1.e-7]
    net_regularizer: 0
    net_dropout: 0.1
    learning_rate: 1.e-3
    batch_size: 8192
    ```

    Note that for challenge task 1, participants can only tune the above five hyper-parameters in `config/DIN_microlens_mmctr_tuner_config_01.yaml`. Other hyper-parameters should be fixed.
    
    We get the best validation AUC: 0.8655.

2. Make predictions on the test set:

    After model training, you can obtain the result file `DIN_microlens_mmctr_tuner_config_01.csv`. Find the best validation AUC from the result csv file, and obtain the corresponding `experiment_id`. Then you can run predictions on the test set.

    ```
    python prediction.py --config config/DIN_microlens_mmctr_tuner_config_01 --expid DIN_MicroLens_1M_x1_xxx --gpu 0
    ```

    After finishing prediction, you can submit the solution file `submission/DIN_MicroLens_1M_x1_xxx.zip`.

3. Make a submission to [the leaderboard](https://www.codabench.org/competitions/5372/#/results-tab).

    <div align="left">
        <img width="90%" src="https://cdn.jsdelivr.net/gh/reczoo/WWW2025_MMCTR_Challenge@main/img/submission_v1.jpg">
    </div>

### Potential Improvements

+ To build the baseline, we simply reuse the DIN model, which is popular for sequential user interest modeling for CTR prediction. We encourage participants to explore some other alternatives for Challenge Task 2.
+ We currently only take extracted text and image embeddings from Bert and CLIP. We encourage participants to explore some new LLMs/MLLMs for multimodal item embedding. Item embedding models can also be trained via sequential modeling or contrastive learning.
+ We only concatenate text and image embeddings and apply PCA for dimensionality reduction. It is interesting to explore other methods for fusing multimodal embedding features.

### Discussion
Welcome to join our WeChat group for any question and discussion. Or you can start a new topic on [the Codabench forum](https://www.codabench.org/forums/5287/).

![Scan QR code](https://cdn.jsdelivr.net/gh/reczoo/WWW2025_MMCTR_Challenge@main/img/wechat.png)
