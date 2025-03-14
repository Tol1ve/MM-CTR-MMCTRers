#!/bin/bash
python run_expid.py --config ./config/DIN_attn_emb_v3 --expid DIN_DIN_attn_emb_v2_001_acf37100 --gpu 0
python prediction.py --config ./config/DIN_attn_emb_v3 --expid DIN_DIN_attn_emb_v2_001_acf37100 --gpu 0