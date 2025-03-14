# # siglip
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# # os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # 启用加速协议

# # 若需代理
# # os.environ['http_proxy'] = 'http://127.0.0.1:10808'

# from transformers import AutoProcessor, AutoModel
# from transformers import AutoProcessor, AutoModelForImageTextToText

# import torch
# from PIL import Image
# import pandas as pd
# import numpy as np

# # # 加载处理器和模型 SIGLIP
# processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
# model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")


# from transformers import AutoProcessor, AutoModelForImageTextToText
# # 移动模型到 GPU（如果可用）
# device = "cuda:1" if torch.cuda.is_available() else "cpu"
# model.to(device)

# # 读取包含文本嵌入的 Parquet 文件
# text_emb_df = pd.read_parquet("/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/2new_item_emb_128.parquet")

# # 获取 txt_emb_BERT 的维度
# text_embedding_dim = len(text_emb_df['txt_emb_BERT'].iloc[0])

# # 图片文件夹路径
# image_folder = "/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/MicroLens_1M_x1/item_images"

# # 获取所有图片文件
# image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

# # 初始化图片嵌入列表
# image_embeddings = []

# # 遍历图片文件
# i = 0
# for image_file in image_files:
#     try:
#         # 打开图片
#         image = Image.open(image_file).convert("RGB")
#         # 预处理图片
#         inputs = processor(images=image, return_tensors="pt")
#         # 添加空的 input_ids
#         inputs["input_ids"] = torch.zeros((1, 1), dtype=torch.long).to(device)

#         inputs = {k: v.to(device) for k, v in inputs.items()}

#         # 提取图片嵌入
#         with torch.no_grad():
#             outputs = model(**inputs)
#             # 尝试不同的方式提取图像特征
#             try:
#                 # 常见的图像特征提取方式
#                 print(f"begin{i}/n")
#                 image_features = outputs.vision_model_output.pooler_output
#             except AttributeError:
#                 try:
#                     # 另一种可能的方式
#                     image_features = outputs.last_hidden_state.mean(dim=1)
#                 except AttributeError:
#                     print(f"无法从 outputs 中提取图像特征: {outputs}")
#                     continue
#             image_features = torch.nn.functional.normalize(image_features, dim=-1)
#             image_emb = image_features.cpu().numpy().flatten()
#             # 调整维度匹配 txt_emb_BERT
#             if len(image_emb) != text_embedding_dim:
#                 if len(image_emb) > text_embedding_dim:
#                     image_emb = image_emb[:text_embedding_dim]
#                 else:
#                     padding = np.zeros(text_embedding_dim - len(image_emb))
#                     image_emb = np.concatenate([image_emb, padding])
#             image_embeddings.append(image_emb)
#             print(f"end{i}/n")
#             i += 1
#     except Exception as e:
#         print(f"Error processing {image_file}: {e}")

# # 确保图片嵌入数量与文本嵌入数量一致（如果有必要）
# if len(image_embeddings) != len(text_emb_df):
#     print("Warning: The number of image embeddings does not match the number of text embeddings.")

# # 将图片嵌入保存到 DataFrame
# image_embedding_df = pd.DataFrame({"image_emb_SIGLIP": image_embeddings})

# # 合并文本嵌入和图片嵌入
# merged_df = pd.concat([text_emb_df.reset_index(drop=True), image_embedding_df.reset_index(drop=True)], axis=1)

# # 保存合并后的 DataFrame 到新的 Parquet 文件
# merged_file_path = "/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/bertAndSiglip_emb.parquet"
# merged_df.to_parquet(merged_file_path)

# print(f"合并后的 Parquet 文件已保存到 {merged_file_path}")




# qwen
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# # os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # 启用加速协议


# from transformers import AutoProcessor, AutoModel
# from transformers import AutoProcessor, AutoModelForImageTextToText

# from tqdm import tqdm
# import torch
# from PIL import Image
# import pandas as pd
# import numpy as np

# # # 加载处理器和模型 QWEN
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
# model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# # 移动模型到 GPU（如果可用）
# device = "cuda:3" if torch.cuda.is_available() else "cpu"
# model.to(device)

# # 读取包含文本嵌入的 Parquet 文件
# text_emb_df = pd.read_parquet("/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/bert_item_emb_128.parquet")

# # 获取 txt_emb_BERT 的维度
# text_embedding_dim = len(text_emb_df['txt_emb_BERT'].iloc[0])

# # 图片文件夹路径
# # image_folder = "/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/MicroLens_1M_x1/item_images"

# # 获取所有图片文件
# # image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg'))])
# # print(len(image_files))
# # 初始化图片嵌入列表
# image_embeddings = []
# image_dir = "/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/MicroLens_1M_x1/item_images"
# # 遍历图片文件
# i = 0
# # 按顺序处理图片
# for idx in tqdm(range(1, 91718)):  # 图片编号从1到91717
#     image_file = os.path.join(image_dir, f"{idx}.jpg")
#     try:
#         # 打开图片
#         image = Image.open(image_file).convert("RGB")
#         # 预处理图片
#         inputs = processor(images=image, return_tensors="pt")
#         # 添加空的 input_ids
#         inputs["input_ids"] = torch.zeros((1, 1), dtype=torch.long).to(device)

#         inputs = {k: v.to(device) for k, v in inputs.items()}

#         # 提取图片嵌入
#         with torch.no_grad():
#             outputs = model(**inputs)
#             # 尝试不同的方式提取图像特征
#             try:
#                 # 常见的图像特征提取方式
#                 print(f"begin{i}/n")
#                 image_features = outputs.vision_model_output.pooler_output
#             except AttributeError:
#                 try:
#                     # 另一种可能的方式
#                     image_features = outputs.last_hidden_state.mean(dim=1)
#                 except AttributeError:
#                     print(f"无法从 outputs 中提取图像特征: {outputs}")
#                     continue
#             image_features = torch.nn.functional.normalize(image_features, dim=-1)
#             image_emb = image_features.cpu().numpy().flatten()
#             # 调整维度匹配 txt_emb_BERT
#             if len(image_emb) != text_embedding_dim:
#                 if len(image_emb) > text_embedding_dim:
#                     image_emb = image_emb[:text_embedding_dim]
#                 else:
#                     padding = np.zeros(text_embedding_dim - len(image_emb))
#                     image_emb = np.concatenate([image_emb, padding])
#             image_embeddings.append(image_emb)
#             print(f"end{i}/n")
#             i += 1
#     except Exception as e:
#         print(f"Error processing {image_file}: {e}")

# # 确保图片嵌入数量与文本嵌入数量一致（如果有必要）
# if len(image_embeddings) != len(text_emb_df):
#     print("Warning: The number of image embeddings does not match the number of text embeddings.")

# # 将图片嵌入保存到 DataFrame
# image_embedding_df = pd.DataFrame({"image_emb_QWEN": image_embeddings})

# # 合并文本嵌入和图片嵌入
# merged_df = pd.concat([text_emb_df.reset_index(drop=True), image_embedding_df.reset_index(drop=True)], axis=1)

# # 保存合并后的 DataFrame 到新的 Parquet 文件
# merged_file_path = "/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/BertAndQwen_emb.parquet"
# merged_df.to_parquet(merged_file_path)

# print(f"合并后的 Parquet 文件已保存到 {merged_file_path}")




# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# from transformers import AutoProcessor, AutoModelForImageTextToText
# from tqdm import tqdm
# import torch
# from PIL import Image
# import pandas as pd
# import numpy as np

# # 加载处理器和模型 QWEN
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
# model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# # 移动模型到 GPU（如果可用）
# device = "cuda:3" if torch.cuda.is_available() else "cpu"
# model.to(device)

# # 图片文件夹路径
# image_folder = "/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/MicroLens_1M_x1/item_images"
# # 输出文件路径
# output_file = "/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/Qwen_emb.parquet"

# # 获取图片文件列表
# image_files = sorted(os.listdir(image_folder))
# embeddings = []

# # 按顺序处理每张图片
# for image_file in tqdm(image_files, desc="Processing images"):
#     image_path = os.path.join(image_folder, image_file)
#     try:
#         # 打开图片
#         image = Image.open(image_path).convert("RGB")
#         # 处理图片
#         inputs = processor(images=image, return_tensors="pt").to(device)
#         # 提取图片特征
#         with torch.no_grad():
#             outputs = model(**inputs)
#             # 这里需要根据模型实际输出确定如何获取embedding，假设取最后一层隐藏状态
#             image_features = outputs.last_hidden_state.mean(dim=1)
#             embedding = image_features.cpu().numpy().flatten()
#             embeddings.append(embedding)
#     except Exception as e:
#         print(f"处理 {image_file} 时出错: {e}")

# # 创建DataFrame
# df = pd.DataFrame({"image_emb_QWEN": embeddings})

# # 打印随机一列的shape
# random_index = df.sample(1).index[0]
# print(f"随机一列的shape: {df.loc[random_index, 'image_emb_QWEN'].shape}")

# # 保存为parquet文件
# df.to_parquet(output_file)




import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # 启用加速协议

# 若需代理
# os.environ['http_proxy'] = 'http://127.0.0.1:10808'

from transformers import AutoProcessor, AutoModel
from transformers import AutoProcessor, AutoModelForImageTextToText

import torch
from PIL import Image
import pandas as pd
import numpy as np

# 加载处理器和模型 QWEN
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")


# 移动模型到 GPU（如果可用）
device = "cuda:3" if torch.cuda.is_available() else "cpu"
model.to(device)

# 读取包含文本嵌入的 Parquet 文件
text_emb_df = pd.read_parquet("/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/bert_item_emb_128.parquet")

# 获取 txt_emb_BERT 的维度
text_embedding_dim = len(text_emb_df['txt_emb_BERT'].iloc[0])

# 图片文件夹路径
image_folder = "/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/MicroLens_1M_x1/item_images"

# 获取所有图片文件
image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

# 初始化图片嵌入列表
image_embeddings = []

# 遍历图片文件
i = 0
for image_file in image_files:
    try:
        # 打开图片
        image = Image.open(image_file)
        if image is None:
            print(f"无法打开图像: {image_file}")
            continue
        image = image.convert("RGB")
        # 预处理图片
        inputs = processor(images=image, return_tensors="pt")
        # 添加空的 input_ids
        inputs["input_ids"] = torch.zeros((1, 1), dtype=torch.long).to(device)

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 提取图片嵌入
        with torch.no_grad():
            outputs = model(**inputs)
            # 尝试不同的方式提取图像特征
            try:
                # 常见的图像特征提取方式
                print(f"begin{i}/n")
                image_features = outputs.vision_model_output.pooler_output
            except AttributeError:
                try:
                    # 另一种可能的方式
                    image_features = outputs.last_hidden_state.mean(dim=1)
                except AttributeError:
                    print(f"无法从 outputs 中提取图像特征: {outputs}")
                    continue
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            image_emb = image_features.cpu().numpy().flatten()
            # 调整维度匹配 txt_emb_BERT
            if len(image_emb) != text_embedding_dim:
                if len(image_emb) > text_embedding_dim:
                    image_emb = image_emb[:text_embedding_dim]
                else:
                    padding = np.zeros(text_embedding_dim - len(image_emb))
                    image_emb = np.concatenate([image_emb, padding])
            image_embeddings.append(image_emb)
            print(f"end{i}/n")
            i += 1
    except Exception as e:
        print(f"Error processing {image_file}: {e}")

# 确保图片嵌入数量与文本嵌入数量一致（如果有必要）
if len(image_embeddings) != len(text_emb_df):
    print("Warning: The number of image embeddings does not match the number of text embeddings.")

# 将图片嵌入保存到 DataFrame
image_embedding_df = pd.DataFrame({"image_emb_Qwen": image_embeddings})

# 合并文本嵌入和图片嵌入
merged_df = pd.concat([text_emb_df.reset_index(drop=True), image_embedding_df.reset_index(drop=True)], axis=1)

# 保存合并后的 DataFrame 到新的 Parquet 文件
merged_file_path = "/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/bertAndQwen_emb.parquet"
merged_df.to_parquet(merged_file_path)

print(f"合并后的 Parquet 文件已保存到 {merged_file_path}")
