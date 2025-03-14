# 769 -> 768
# import pandas as pd

# # 读取 Parquet 文件
# file_path = '/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/BertAndSiglip_item_emb_2column_769.parquet'
# df = pd.read_parquet(file_path)

# # 删除除列名外的第一行数据
# df = df[1:]

# # 重置索引
# df = df.reset_index(drop=True)

# # 保存为新的 Parquet 文件
# new_file_path = '/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/BertAndSiglip_item_emb_2column_768.parquet'
# df.to_parquet(new_file_path)

# print(f"新文件已保存到 {new_file_path}")





# 两列分别都降到128维
# import pandas as pd
# from sklearn.decomposition import PCA

# # 读取 Parquet 文件
# file_path = '/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/BertAndSiglip_item_emb_2column_768.parquet'
# df = pd.read_parquet(file_path)

# # 初始化 PCA 模型，指定降维到 128 维
# pca = PCA(n_components=128)

# # 对每一列分别进行降维处理
# for col in df.columns:
#     # 提取当前列的数据
#     col_data = df[col].tolist()
#     # 执行 PCA 降维
#     reduced_col_data = pca.fit_transform(col_data)
#     # 将降维后的数据转换为 DataFrame 列
#     df[col] = pd.Series(reduced_col_data.tolist())

# # 保存降维后的数据为新的 Parquet 文件
# new_file_path = '/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/BertAndSiglip_item_emb_2column_128.parquet'
# df.to_parquet(new_file_path)

# print(f"降维后的数据已保存到 {new_file_path}")




#两列都降到64维然后拼成一列128维
# import pandas as pd
# import numpy as np
# from sklearn.decomposition import PCA

# # 读取 Parquet 文件
# file_path = '/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/BertAndSiglip_item_emb_2column_768.parquet'
# df = pd.read_parquet(file_path)

# # 提取两列数据
# col1 = np.array(df.iloc[:, 0].tolist())
# col2 = np.array(df.iloc[:, 1].tolist())

# # 将两列数据合并
# combined_data = np.concatenate((col1, col2), axis=1)

# # 第一步：将两列数据降维成一列
# pca_1 = PCA(n_components=768)
# reduced_data_1 = pca_1.fit_transform(combined_data)

# # 第二步：将这一列数据的每个元素都降维成 128 维
# pca_2 = PCA(n_components=128)
# reduced_data_2 = pca_2.fit_transform(reduced_data_1)

# # 将降维后的数据转换为列表
# reduced_list = [list(row) for row in reduced_data_2]

# # 创建新的 DataFrame
# new_df = pd.DataFrame({'BertAndSiglip_item_emb_v1': reduced_list})

# # 保存为新的 Parquet 文件
# output_path = '/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/BertAndSiglip_item_emb_128_v1.parquet'
# new_df.to_parquet(output_path)

# print(f"处理后的数据已保存到 {output_path}")



# 分别处理为64维，然后合并为128维
# import pandas as pd
# import numpy as np
# from sklearn.decomposition import PCA

# # 读取 Parquet 文件
# file_path = '/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/BertAndSiglip_item_emb_2column_768.parquet'
# df = pd.read_parquet(file_path)

# # 提取两列数据并转换为 numpy 数组
# col1 = np.array(df.iloc[:, 0].tolist())
# col2 = np.array(df.iloc[:, 1].tolist())

# # 分别对两列数据进行 PCA 降维到 64 维
# pca = PCA(n_components=64)
# col1_reduced = pca.fit_transform(col1)
# col2_reduced = pca.fit_transform(col2)

# # 合并降维后的两列数据
# combined_reduced = np.hstack((col1_reduced, col2_reduced))

# # 将合并后的数据转换为列表
# combined_list = [list(row) for row in combined_reduced]

# # 创建新的 DataFrame
# new_df = pd.DataFrame({'BertAndSiglip_item_emb_v2': combined_list})

# # 保存为新的 Parquet 文件
# output_path = '/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/BertAndSiglip_item_emb_128_v2.parquet'
# new_df.to_parquet(output_path)

# print(f"处理后的数据已保存到 {output_path}")




import pandas as pd
from sklearn.decomposition import PCA

# 读取 Parquet 文件
file_path = '/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/BertAndSiglip_item_emb_2column_768.parquet'
df = pd.read_parquet(file_path)

# 初始化 PCA 模型，指定降维到 128 维
pca = PCA(n_components=128)

# 对 txt_emb_BERT 列进行降维处理
txt_col_data = df['txt_emb_BERT'].tolist()
reduced_txt_col_data = pca.fit_transform(txt_col_data)
txt_df = pd.DataFrame({'txt_emb_BERT': pd.Series(reduced_txt_col_data.tolist())})

# 保存降维后的 txt_emb_BERT 列数据为新的 Parquet 文件
txt_file_path = '/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/txt_emb_BERT_128.parquet'
txt_df.to_parquet(txt_file_path)

# 对 image_emb_SIGLIP 列进行降维处理
image_col_data = df['image_emb_SIGLIP'].tolist()
reduced_image_col_data = pca.fit_transform(image_col_data)
image_df = pd.DataFrame({'image_emb_SIGLIP': pd.Series(reduced_image_col_data.tolist())})

# 保存降维后的 image_emb_SIGLIP 列数据为新的 Parquet 文件
image_file_path = '/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/image_emb_SIGLIP_128.parquet'
image_df.to_parquet(image_file_path)

print(f"降维后的 txt_emb_BERT 数据已保存到 {txt_file_path}")
print(f"降维后的 image_emb_SIGLIP 数据已保存到 {image_file_path}")
    