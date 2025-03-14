# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pandas as pd
import torch


class ParquetDataset(Dataset):
    def __init__(self, data_path):
        self.column_index = dict()
        self.darray = self.load_data(data_path)
        
    def __getitem__(self, index):
        return self.darray[index, :]
    
    def __len__(self):
        return self.darray.shape[0]

    def load_data(self, data_path):
        df = pd.read_parquet(data_path)
        data_arrays = []
        idx = 0
        for col in df.columns:
            if df[col].dtype == "object":
                array = np.array(df[col].to_list())
                seq_len = array.shape[1]
                self.column_index[col] = [i + idx for i in range(seq_len)]
                idx += seq_len
            else:
                array = df[col].to_numpy()
                self.column_index[col] = idx
                idx += 1
            data_arrays.append(array)
        return np.column_stack(data_arrays)

 
class MMCTRDataLoader(DataLoader):
    def __init__(self, feature_map, data_path, item_info, batch_size=32, shuffle=False,
                 num_workers=1, max_len=100, **kwargs):
        if not data_path.endswith(".parquet"):
            data_path += ".parquet"
        self.dataset = ParquetDataset(data_path)
        column_index = self.dataset.column_index
        super().__init__(dataset=self.dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers,
                         collate_fn=BatchCollator(feature_map, max_len, column_index, item_info))
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self):
        return self.num_batches
    
class LYDataLoader(DataLoader):
    def __init__(self, feature_map, data_path, item_info, batch_size=32, shuffle=False,
                 num_workers=1, max_len=100, **kwargs):
        if not data_path.endswith(".parquet"):
            data_path += ".parquet"
        self.dataset = ParquetDataset(data_path)
        column_index = self.dataset.column_index
        super().__init__(dataset=self.dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers,
                         collate_fn=BatchCollator_from_multimodel(feature_map, max_len, column_index, item_info))
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self):
        return self.num_batches


class BatchCollator(object):
    def __init__(self, feature_map, max_len, column_index, item_info,):
        self.feature_map = feature_map
        self.item_info = pd.read_parquet(item_info)
        self.max_len = max_len
        self.column_index = column_index
    def like_level(self,item_dict,batch_items):
        self.item_feature_info=pd.read_parquet('./data/item_feature_new.parquet')
        ll_info= self.item_feature_info.iloc[batch_items]
        item_dict["likes_level"] = torch.from_numpy(np.array(ll_info['likes_level'].to_list()))
        item_dict["views_level"] = torch.from_numpy(np.array(ll_info['views_level'].to_list()))
        return item_dict
    def __call__(self, batch):
        batch_tensor = default_collate(batch)
        all_cols = set(list(self.feature_map.features.keys()) + self.feature_map.labels)
        batch_dict = dict()
        for col, idx in self.column_index.items():
            if col in all_cols:
                batch_dict[col] = batch_tensor[:, idx]
        batch_seqs = batch_dict["item_seq"][:, -self.max_len:]
        del batch_dict["item_seq"]
        mask = (batch_seqs > 0).float() # zeros for masked positions
        item_index = batch_dict["item_id"].numpy().reshape(-1, 1)
        del batch_dict["item_id"]
        batch_items = np.hstack([batch_seqs.numpy(), item_index]).flatten()
        item_info = self.item_info.iloc[batch_items]
        item_dict = dict()
        for col in item_info.columns:
            if col in all_cols:
                item_dict[col] = torch.from_numpy(np.array(item_info[col].to_list()))
        ##mod
        item_dict=self.like_level(item_dict,batch_items)
        #batch_dict指
        return batch_dict, item_dict, mask
    
    
    
class BatchCollator_from_multimodel(object):
    def __init__(self, feature_map, max_len, column_index, item_info,multimodel_info='./data/item_emb.parquet'):
        self.feature_map = feature_map
        self.item_info = pd.read_parquet(item_info)
        self.max_len = max_len
        self.column_index = column_index
        if multimodel_info:
            item2=pd.read_parquet(multimodel_info)
            def combine(item_origin,item_addition):
                emb0 = np.stack(item_origin["item_emb_d128"].values)
                emb3 = np.stack(item_addition["item_emb_d128_v3"].values)
                zero_vector = np.zeros_like(emb3[0])

                emb4=np.stack(item_addition["item_emb_d128_e4"].values)
                emb1=np.stack(item_addition["item_emb_d128_v1"].values)
                emb2=np.stack(item_addition["item_emb_d128_v2"].values)
                embs=[emb0,emb1,emb2,emb3,emb4]
                for i, emb in enumerate(embs):
                    if i==0:
                        continue
                    embs[i] = np.vstack([zero_vector, emb3])
                weight=[0.4,0.1,0,0.4,0.1]
                fused_emb=np.zeros_like(emb0)
                for i in range(len(embs)):
                    fused_emb+=embs[i]*weight[i]
                item_origin["item_emb_d128"] = list(fused_emb)
                return item_origin
            self.item_info=combine(self.item_info,item2)
    def like_level(self,item_dict,batch_items):
        self.item_feature_info=pd.read_parquet('/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/item_feature_new.parquet')
        ll_info= self.item_feature_info.iloc[batch_items]
        item_dict["likes_level"] = torch.from_numpy(np.array(ll_info['likes_level'].to_list()))
        item_dict["views_level"] = torch.from_numpy(np.array(ll_info['views_level'].to_list()))
        return item_dict
    def __call__(self, batch):
        batch_tensor = default_collate(batch)
        all_cols = set(list(self.feature_map.features.keys()) + self.feature_map.labels)
        batch_dict = dict()
        for col, idx in self.column_index.items():
            if col in all_cols:
                batch_dict[col] = batch_tensor[:, idx]
        batch_seqs = batch_dict["item_seq"][:, -self.max_len:]
        del batch_dict["item_seq"]
        mask = (batch_seqs > 0).float() # zeros for masked positions
        item_index = batch_dict["item_id"].numpy().reshape(-1, 1)
        del batch_dict["item_id"]
        batch_items = np.hstack([batch_seqs.numpy(), item_index]).flatten()
        item_info = self.item_info.iloc[batch_items]
        item_dict = dict()
        for col in item_info.columns:
            if col in all_cols:
                item_dict[col] = torch.from_numpy(np.array(item_info[col].to_list()))
        ##mod
        item_dict=self.like_level(item_dict,batch_items)
        #batch_dict指
        return batch_dict, item_dict, mask
    
    



class nagYDataLoader(DataLoader):
    def __init__(self, feature_map, data_path, item_info, batch_size=32, shuffle=False,
                 num_workers=1, max_len=100, **kwargs):
        if not data_path.endswith(".parquet"):
            data_path += ".parquet"
        self.dataset = ParquetDataset(data_path)
        column_index = self.dataset.column_index
        super().__init__(dataset=self.dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers,
                         collate_fn=YBatchCollator(feature_map, max_len, column_index, item_info))
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self):
        return self.num_batches
    
    
class YBatchCollator(object):
    def __init__(self, feature_map, max_len, column_index, item_info, multimodel_info='./data/item_emb.parquet', multimodel_info2='./data/BertAndSiglip_item_emb_128_v1.parquet', multimodel_info3='./data/BertAndSiglip_item_emb_128_v2.parquet', multimodel_info4='./data/txt_emb_BERT_128.parquet', multimodel_info5='./data/image_emb_SIGLIP_128.parquet'):
        self.feature_map = feature_map
        self.item_info = pd.read_parquet(item_info)
        self.max_len = max_len
        self.column_index = column_index
        if multimodel_info:
            item2=pd.read_parquet(multimodel_info)
            item3=pd.read_parquet(multimodel_info2)
            item4=pd.read_parquet(multimodel_info3)
            item5=pd.read_parquet(multimodel_info4)
            item6=pd.read_parquet(multimodel_info5)
            def combine(item_info,item2,item3,item4):
                emb0 = np.stack(item_info["item_emb_d128"].values)
                emb3 = np.stack(item2["item_emb_d128_v3"].values)
                zero_vector = np.zeros_like(emb3[0])

                emb4=np.stack(item2["item_emb_d128_e4"].values)
                emb1=np.stack(item2["item_emb_d128_v1"].values)
                emb2=np.stack(item2["item_emb_d128_v2"].values)
                emb5=np.stack(item3["BertAndSiglip_item_emb_v1"].values)
                emb6=np.stack(item4["BertAndSiglip_item_emb_v2"].values)
                emb7=np.stack(item5["txt_emb_BERT"].values)
                emb8=np.stack(item6["image_emb_SIGLIP"].values)
                embs=[emb0,emb1,emb2,emb3,emb4,emb5,emb6,emb7,emb8]
                for i, emb in enumerate(embs):
                    if i==0:
                        continue
                    embs[i] = np.vstack([zero_vector, emb3])
                weight=[0.5,0.,0.,0.,0.,0.,0.,0.3,0.2]
                fused_emb=np.zeros_like(emb0)
                for i in range(len(embs)):
                    fused_emb+=embs[i]*weight[i]
                item_info["item_emb_d128"] = list(fused_emb)
                return item_info
            self.item_info=combine(self.item_info,item2,item3,item4)
    def like_level(self,item_dict,batch_items):
        self.item_feature_info=pd.read_parquet('/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/item_feature_new.parquet')
        ll_info= self.item_feature_info.iloc[batch_items]
        item_dict["likes_level"] = torch.from_numpy(np.array(ll_info['likes_level'].to_list()))
        item_dict["views_level"] = torch.from_numpy(np.array(ll_info['views_level'].to_list()))
        return item_dict
    def __call__(self, batch):
        batch_tensor = default_collate(batch)
        all_cols = set(list(self.feature_map.features.keys()) + self.feature_map.labels)
        batch_dict = dict()
        for col, idx in self.column_index.items():
            if col in all_cols:
                batch_dict[col] = batch_tensor[:, idx]
        batch_seqs = batch_dict["item_seq"][:, -self.max_len:]
        del batch_dict["item_seq"]
        mask = (batch_seqs > 0).float() # zeros for masked positions
        item_index = batch_dict["item_id"].numpy().reshape(-1, 1)
        del batch_dict["item_id"]
        batch_items = np.hstack([batch_seqs.numpy(), item_index]).flatten()
        item_info = self.item_info.iloc[batch_items]
        item_dict = dict()
        for col in item_info.columns:
            if col in all_cols:
                item_dict[col] = torch.from_numpy(np.array(item_info[col].to_list()))
        ##mod
        item_dict=self.like_level(item_dict,batch_items)
        #batch_dict指
        return batch_dict, item_dict, mask


class YDataLoader(DataLoader):
    def __init__(self, feature_map, data_path, item_info, batch_size=32, shuffle=False,
                 num_workers=1, max_len=100, **kwargs):
        if not data_path.endswith(".parquet"):
            data_path += ".parquet"
        self.dataset = ParquetDataset(data_path)
        column_index = self.dataset.column_index
        super().__init__(dataset=self.dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers,
                         collate_fn=YBatchCollator(feature_map, max_len, column_index, item_info))
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self):
        return self.num_batches



if __name__ == '__main__':
    a = MMCTRDataLoader(None, '/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/MicroLens_1M_x1/train.parquet', '/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/MicroLens_1M_x1/item_info.parquet')