import numpy as np
import polars as pl
a = pl.read_parquet("/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/item_feature_new.parquet")
new_row = pl.DataFrame({
    "item_id":[0],
    "item_title": [''],  # 假设第一列是字符串类型
    "item_tags": [[0]],   # 假设是一个列表列
    "likes_level": [0],    # 假设是一个整数列
    "views_level": [0],    # 假设是一个整数列
    "txt_emb_BERT": [np.zeros(768, dtype=np.float32) ], # 假设是一个数组列，大小为 768
    "img_emb_CLIPRN50": [np.zeros(1024, dtype=np.float32)]  # 假设是一个数组列，大小为 1024
})
a = pl.concat([new_row, a], how="vertical")
a.write_parquet("/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/item_feature_new.parquet")