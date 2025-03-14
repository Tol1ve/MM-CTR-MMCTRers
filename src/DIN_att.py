
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

import torch
from torch import nn
import torch.nn.functional as F
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, DIN_Attention, Dice
from fuxictr.utils import not_in_whitelist
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.attention import MultiHeadAttention

class CrossAttention(nn.Module):
    def __init__(self, 
                 embedding_dim=64,
                 num_heads=4,
                 dropout=0.1,
                 use_softmax=True):
        super(CrossAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.use_softmax = use_softmax
        
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, target_item, history_sequence, mask=None):
        batch_size, seq_len, _ = history_sequence.size()
        
        # Prepare Q/K/V
        query = self.query_proj(target_item)
        if len(target_item.shape) == 2:
            query = query.unsqueeze(0)
        elif len(target_item.shape) == 3:
            query = query.transpose(0, 1)
        key = self.key_proj(history_sequence.transpose(0, 1))
        value = self.value_proj(history_sequence.transpose(0, 1)) 
        
        # Create attention mask
        if mask is not None:
            attn_mask = (~mask.bool())
        else:
            attn_mask = None
        
        # Cross-attention
        attn_output, _ = self.multihead_attn(query=query,
                                             key=key,
                                             value=value,
                                             key_padding_mask=attn_mask)
        
        # Process output

        if attn_output.shape[0] ==1:
             attn_output = attn_output.squeeze(0)
        else:
            attn_output = attn_output.transpose(0, 1)
        attn_output = self.norm(attn_output)
        return attn_output

    
class CrossTransformerLayer(nn.Module):
    def __init__(self, 
                 embedding_dim=256,
                 num_heads=4,
                 ff_dim=256,
                 dropout=0.1,
                 use_softmax=True):
        super(CrossTransformerLayer, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Cross-Attention Block
        self.cross_attn = CrossAttention(embedding_dim, num_heads, dropout, use_softmax)
        self.attn_norm = nn.LayerNorm(embedding_dim)
        
        # Feed-Forward Block
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim)
        )
        self.ffn_norm = nn.LayerNorm(embedding_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, target_item, history_sequence, mask=None):
        # Cross-Attention with residual connection
        attn_out = self.cross_attn(target_item, history_sequence, mask)
        attn_out = self.attn_norm(target_item + self.dropout(attn_out))
        
        # Feed-Forward with residual connection
        ffn_out = self.ffn(attn_out)
        output = self.ffn_norm(attn_out + self.dropout(ffn_out))
        
        return output

class CrossTransformerEncoder(nn.Module):
    def __init__(self,
                 num_layers=2,
                 embedding_dim=256,
                 num_heads=4,
                 ff_dim=256,
                 dropout=0.1,
                 use_softmax=True):
        super(CrossTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            CrossTransformerLayer(embedding_dim, num_heads, ff_dim, dropout, use_softmax)
            for _ in range(num_layers)
        ])
        self.USE_self_attn = False

    def forward(self, target_item, history_sequence, mask=None):
        Query = target_item
        for i,layer in enumerate(self.layers):
            if(i==0 and self.USE_self_attn):
                output = layer(history_sequence, history_sequence, mask)
            elif(i==0):
                output = layer(Query, history_sequence,mask)
            else:
                output = layer(output, history_sequence, mask)
        return output



class DIN_att(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DIN", 
                 gpu=0, 
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 learning_rate=1e-3, 
                 embedding_dim=64, 
                 net_dropout=0, 
                 batch_norm=False, 
                 din_use_softmax=False,
                 accumulation_steps=1,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(DIN_att, self).__init__(feature_map,
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        if isinstance(dnn_activations, str) and dnn_activations.lower() == "dice":
            dnn_activations = [Dice(units) for units in dnn_hidden_units]
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim # 64
        self.item_info_dim = 0  # 所有source为item的特征的embedding_dim之和 64+64+128=256
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim)
        ##mod
        self.alpha = 0.5
        self.pos_weight = 1
        self.item_info_dim +=128
        self.accumulation_steps = accumulation_steps
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.attention_layers = DIN_Attention(
                                            self.item_info_dim,
                                            attention_units=attention_hidden_units,
                                            hidden_activations=attention_hidden_activations,
                                            output_activation=attention_output_activation,
                                            dropout_rate=attention_dropout,
                                            use_softmax=din_use_softmax)
 
        self.cross_attention_layers = CrossTransformerEncoder(num_layers = 2,
                                                    embedding_dim = self.item_info_dim,
                                                     num_heads=4,
                                                     ff_dim=self.item_info_dim,
                                                     dropout=attention_dropout,
                                                     use_softmax=True)

        input_dim = feature_map.sum_emb_out_dim() + self.item_info_dim*2+128    # 384+256
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        '''
        inputs:
            batch_dict = {likes_level: [batch_size],
                      view_level: [batch_size] }
            item_dict = {item_id: [batch_size*(64+1)],
                     item_tags: [batch_size*(64+1),5],
                     item_emb_d128: [batch_size*(64+1),128]}
            mask = [batch_size, 64]  
        '''
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        emb_list = []
        if batch_dict: # not empty
            feature_emb = self.embedding_layer(batch_dict, flatten_emb=True)
            emb_list.append(feature_emb)   # torch.Size([batch_size, (64+64)]) likes_level, view_level两个特征
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)  # torch.Size([batch_size*65, (64+64+128)]) ,item_id, item_tags, item_emb_d128三个特征
        batch_size = mask.shape[0]   # mask: [batch_size, 64]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)  # torch.Size([batch_size, 65, 256])
        target_emb = item_feat_emb[:, -1, :]  # torch.Size([batch_size, 256])
        sequence_emb = item_feat_emb[:, 0:-1, :]  # torch.Size([batch_size, 64, 256])
        
        pooling_emb = self.attention_layers(target_emb, sequence_emb, mask)
        
        pooling_emb_cross = self.cross_attention_layers(target_emb, sequence_emb, mask)
        
        emb_list += [target_emb, pooling_emb, pooling_emb_cross]  # [feature_emb, target_emb, pooling_emb]  [(batch_size, 128), (batch_size, 256), (batch_size, 256)]
        feature_emb = torch.cat(emb_list, dim=-1) # torch.Size([batch_size, 512])
        logit = self.dnn(feature_emb)
        y_pred = self.output_activation( logit)
        return_dict = {"y_pred": y_pred, "logit": logit}
        return return_dict


    def get_inputs(self, inputs, feature_source=None):
        '''
        用户侧特征的智能过滤 + 物品侧特征的全保留 + 设备自动迁移
        过滤标签特征,过滤meta数据特征（user_id/item_seq）,保留source
        inputs:
            batch_dict = {user_id: [batch_size],
                      likes_level: [batch_size],
                      view_level: [batch_size] }
            item_dict = {item_id: [batch_size*65],
                     item_tags: [batch_size*65,5],
                     item_emb_d128: [batch_size*65,128]}
            mask = [batch_size, 64]       
        '''
        batch_dict, item_dict, mask = inputs

        X_dict = dict()
        for feature, value in batch_dict.items():
            if feature in self.feature_map.labels:
                continue
            feature_spec = self.feature_map.features[feature]
            if feature_spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            X_dict[feature] = value.to(self.device)
        for item, value in item_dict.items():
            item_dict[item] = value.to(self.device)
        return X_dict, item_dict, mask.to(self.device)

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        batch_dict = inputs[0]
        y = batch_dict[labels[0]].to(self.device)
        return y.float().view(-1, 1)
                
    def get_group_id(self, inputs):
        return inputs[0][self.feature_map.group_id]

    def train_step(self, batch_data):
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss = loss / self.accumulation_steps
        loss.backward()
        if (self._batch_index + 1) % self.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss
