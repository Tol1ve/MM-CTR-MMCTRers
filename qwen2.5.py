import pandas as pd
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import os

# 配置参数
class Config:
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    image_dir = "/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/MicroLens_1M_x1/item_images"
    output_dir = "/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data"
    batch_size = 4  # 根据GPU显存调整
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# 设备初始化
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和处理器（模仿示例风格）
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    Config.model_name,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"  # 加速推理
).eval()

processor = AutoProcessor.from_pretrained(Config.model_name)

# 构建消息模板函数
def build_feature_messages(image_path):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "<img></img>"}  # 特征提取专用指令
        ]
    }]

# 批处理特征提取
def extract_features_batch(image_paths):
    try:
        # 构建消息批次
        batch_messages = [build_feature_messages(p) for p in image_paths]
        
        # 处理输入（模仿示例流程）
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) 
                for m in batch_messages]
        image_inputs, _ = process_vision_info(batch_messages)
        
        # 编码输入
        inputs = processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device, dtype=Config.dtype)

        # 前向传播
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=Config.dtype):
            outputs = model(**inputs, output_hidden_states=True)
        
        # 提取视觉特征（根据最新文档调整）
        visual_features = outputs.vision_last_hidden_state.mean(dim=1)
        return visual_features.cpu().numpy()
    
    except Exception as e:
        print(f"批处理失败: {str(e)}")
        return None

# 主处理流程
def main():
    # 加载数据
    bert_df = pd.read_parquet("/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/bert_item_emb_128.parquet")
    siglip_df = pd.read_parquet("/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/BertAndSiglip_item_emb_128_v1.parquet")

    # 获取图片路径列表
    image_paths = [os.path.join(Config.image_dir, f"{i}.jpg") for i in range(1, 91718)]
    
    # 分批处理
    features = []
    for i in tqdm(range(0, len(image_paths), Config.batch_size), desc="Processing"):
        batch_paths = image_paths[i:i+Config.batch_size]
        batch_feats = extract_features_batch(batch_paths)
        
        if batch_feats is not None:
            features.extend(batch_feats)
        else:  # 失败时回退到单张处理
            for p in batch_paths:
                try:
                    feat = extract_features_batch([p])[0]
                    features.append(feat)
                except:
                    features.append(np.zeros(128))  # 零填充
    
    # 合并数据
    bert_df["img_emb_Qwen"] = features
    siglip_df["img_emb_Qwen"] = features
    
    # 保存结果
    bert_df.to_parquet(os.path.join(Config.output_dir, "bert_with_qwen.parquet"))
    siglip_df.to_parquet(os.path.join(Config.output_dir, "siglip_with_qwen.parquet"))

if __name__ == "__main__":
    main()