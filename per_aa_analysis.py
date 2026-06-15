#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-Amino-Acid 准确率分析
加载 best_combined_model.pth，在测试集上逐氨基酸统计密码子预测准确率
"""
import os
os.environ.setdefault('USERNAME', 'user')
os.environ.setdefault('HOME', 'C:/Users/lingongqi')
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import json, itertools, random, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from pathlib import Path

WORK_DIR = Path(__file__).parent

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==================== 配置 ====================
MAX_LEN    = 500
HIDDEN_DIM = 128
BATCH_SIZE = 32
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 密码子信息 ====================
AA_TO_CODONS = {
    'A': ['GCT','GCC','GCA','GCG'], 'R': ['CGT','CGC','CGA','CGG','AGA','AGG'],
    'N': ['AAT','AAC'], 'D': ['GAT','GAC'], 'C': ['TGT','TGC'],
    'Q': ['CAA','CAG'], 'E': ['GAA','GAG'], 'G': ['GGT','GGC','GGA','GGG'],
    'H': ['CAT','CAC'], 'I': ['ATT','ATC','ATA'],
    'L': ['TTA','TTG','CTT','CTC','CTA','CTG'], 'K': ['AAA','AAG'],
    'M': ['ATG'], 'F': ['TTT','TTC'], 'P': ['CCT','CCC','CCA','CCG'],
    'S': ['TCT','TCC','TCA','TCG','AGT','AGC'], 'T': ['ACT','ACC','ACA','ACG'],
    'W': ['TGG'], 'Y': ['TAT','TAC'], 'V': ['GTT','GTC','GTA','GTG'],
}
ALL_CODONS = [''.join(c) for c in itertools.product('ATGC', repeat=3)]
CODON_TO_IDX = {c: i for i, c in enumerate(ALL_CODONS)}
AA_TO_IDX = {aa: i for i, aa in enumerate('ARNDCQEGHILKFPSWYVM')}
HYDROPHOBIC = {'A','V','I','L','M','F','W','Y'}
CONSERVED   = {'C','G','P','W'}

ECOLI_FREQ = {
    'GCT':0.18,'GCC':0.27,'GCA':0.23,'GCG':0.32,
    'CGT':0.22,'CGC':0.28,'CGA':0.05,'CGG':0.05,'AGA':0.04,'AGG':0.04,
    'AAT':0.49,'AAC':0.51,'GAT':0.63,'GAC':0.37,
    'TGT':0.46,'TGC':0.54,'CAA':0.34,'CAG':0.66,
    'GAA':0.68,'GAG':0.32,'GGT':0.25,'GGC':0.25,'GGA':0.15,'GGG':0.35,
    'CAT':0.47,'CAC':0.53,'ATT':0.51,'ATC':0.25,'ATA':0.24,
    'TTA':0.14,'TTG':0.13,'CTT':0.12,'CTC':0.11,'CTA':0.04,'CTG':0.52,
    'AAA':0.76,'AAG':0.24,'TTT':0.58,'TTC':0.42,
    'CCT':0.18,'CCC':0.12,'CCA':0.20,'CCG':0.50,
    'TCT':0.12,'TCC':0.15,'TCA':0.14,'TCG':0.15,'AGT':0.09,'AGC':0.28,
    'ACT':0.18,'ACC':0.40,'ACA':0.12,'ACG':0.30,
    'TGG':1.00,'TAT':0.59,'TAC':0.41,
    'GTT':0.18,'GTC':0.15,'GTA':0.11,'GTG':0.56,'ATG':1.00
}

# ==================== 特征提取 ====================
def extract_features(data):
    print("特征提取（52维）...", flush=True)
    X = np.zeros((len(data), MAX_LEN, 52), dtype=np.float32)
    for idx, item in enumerate(data):
        protein = item['protein']
        prot_len = len(protein)
        for j, aa in enumerate(protein[:MAX_LEN]):
            feat = np.zeros(52, dtype=np.float32)
            ai = AA_TO_IDX.get(aa, 0)
            feat[ai] = 1.0
            if aa in ('A','L','E','M','Q','K','R'): feat[20:23] = [1,0,0]
            elif aa in ('V','I','Y','F','W'):       feat[20:23] = [0,1,0]
            else:                                    feat[20:23] = [0,0,1]
            feat[23] = 0.3 if aa in HYDROPHOBIC else 0.7
            feat[24] = 0.8 if aa in CONSERVED else 0.4
            rel_pos = j / max(prot_len, 1)
            feat[25] = rel_pos
            feat[26] = np.sin(rel_pos * 2 * np.pi)
            aa_codons = AA_TO_CODONS.get(aa, [])
            for k, codon in enumerate(aa_codons[:5]):
                feat[27 + k] = ECOLI_FREQ.get(codon, 0.0)
            X[idx, j] = feat
    return X

def extract_labels(data):
    print("提取标签...", flush=True)
    y = np.zeros((len(data), MAX_LEN), dtype=np.int64)
    for idx, item in enumerate(data):
        for j, codon in enumerate(item['codons'][:MAX_LEN]):
            if codon in CODON_TO_IDX:
                y[idx, j] = CODON_TO_IDX[codon]
    return y

# ==================== 数据集 ====================
class CodonDataset(Dataset):
    def __init__(self, X, y): self.X = torch.FloatTensor(X); self.y = torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ==================== 模型 ====================
class BackTranslationModel(nn.Module):
    def __init__(self, input_dim=52, hidden_dim=HIDDEN_DIM, num_codons=64, num_heads=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = nn.MultiheadAttention(hidden_dim*2, num_heads, dropout=0.3, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim*2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_codons))
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return self.fc(lstm_out + attn_out)

# ==================== 主函数 ====================
def main():
    print("=" * 60, flush=True)
    print("Per-Amino-Acid 准确率分析", flush=True)
    print("=" * 60, flush=True)

    # 1. 加载数据
    json_path = WORK_DIR / 'ecoli_real_cds_combined.json'
    print(f"\n[1/4] 加载数据: {json_path}", flush=True)
    with open(json_path, encoding='utf-8') as f:
        raw = json.load(f)
    data = [r for r in raw if 50 <= r['aa_len'] <= MAX_LEN]
    print(f"  加载 {len(data)} 条", flush=True)

    # 2. 特征 & 标签
    X = extract_features(data)
    y = extract_labels(data)

    # 3. 划分数据集（与训练时相同的随机种子）
    print(f"\n[2/4] 划分数据集...", flush=True)
    indices = list(range(len(data)))
    train_val_idx, test_idx = train_test_split(indices, test_size=1264, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=1264, random_state=42)

    X_test = X[test_idx]
    y_test = y[test_idx]
    test_data = [data[i] for i in test_idx]
    print(f"  测试集: {len(X_test)} 条", flush=True)

    # 4. 加载模型
    print(f"\n[3/4] 加载模型...", flush=True)
    model = BackTranslationModel()
    model.load_state_dict(torch.load(WORK_DIR / 'best_combined_model.pth', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"  模型已加载", flush=True)

    # 5. 推理 + 逐氨基酸统计
    print(f"\n[4/4] 逐氨基酸分析...", flush=True)
    test_loader = DataLoader(CodonDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # 初始化统计
    aa_stats = {}
    for aa in 'ARNDCQEGHILKFPSWYVM':
        aa_stats[aa] = {'correct': 0, 'total': 0, 'pred_codons': Counter(), 'true_codons': Counter()}

    global_idx = 0
    with torch.no_grad():
        for batch_idx, (Xb, yb) in enumerate(test_loader):
            out = model(Xb.to(DEVICE)).argmax(-1).cpu().numpy()

            for i in range(len(out)):
                protein = test_data[global_idx]['protein']
                codons = test_data[global_idx]['codons']
                pred_row = out[i]
                true_row = yb[i].numpy()

                for j in range(len(protein[:MAX_LEN])):
                    if j < len(true_row) and true_row[j] != 0:
                        aa = protein[j]
                        pred_codon_idx = pred_row[j]
                        true_codon_idx = true_row[j]

                        if aa in aa_stats:
                            aa_stats[aa]['total'] += 1
                            if pred_codon_idx == true_codon_idx:
                                aa_stats[aa]['correct'] += 1
                            aa_stats[aa]['pred_codons'][ALL_CODONS[pred_codon_idx]] += 1
                            aa_stats[aa]['true_codons'][ALL_CODONS[true_codon_idx]] += 1

                global_idx += 1

    # 6. 结果分析
    print(f"\n{'='*60}", flush=True)
    print("Per-Amino-Acid 准确率分析", flush=True)
    print(f"{'='*60}\n", flush=True)

    print(f"{'AA':>3} | {'准确率':>8} | {'样本数':>8} | {'真实主密码子':>20} | {'预测主密码子':>20}", flush=True)
    print("-" * 90, flush=True)

    results = []
    for aa in 'ARNDCQEGHILKFPSWYVM':
        stats = aa_stats[aa]
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total']
            true_main = stats['true_codons'].most_common(1)
            pred_main = stats['pred_codons'].most_common(1)
            true_str = f"{true_main[0][0]}({true_main[0][1]})" if true_main else "N/A"
            pred_str = f"{pred_main[0][0]}({pred_main[0][1]})" if pred_main else "N/A"
            results.append((aa, acc, stats['total'], true_str, pred_str))
        else:
            results.append((aa, 0.0, 0, "N/A", "N/A"))

    # 按准确率排序（从低到高）
    results.sort(key=lambda x: x[1])

    for aa, acc, total, true_str, pred_str in results:
        if total > 0:
            marker = " ★" if acc >= 0.7 else (" ⚠" if acc < 0.5 else "  ")
            print(f"{aa:>3} | {acc:>7.2%} | {total:>8,} | {true_str:>20} | {pred_str:>20}{marker}", flush=True)
        else:
            print(f"{aa:>3} |   N/A   | {total:>8} | {true_str:>20} | {pred_str:>20}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("瓶颈氨基酸 (准确率 < 50%):", flush=True)
    bottleneck = [(aa, acc, total) for aa, acc, total, _, _ in results if 0 < acc < 0.5]
    if bottleneck:
        for aa, acc, total in bottleneck:
            print(f"  {aa}: {acc:.2%} ({total:,} 样本)", flush=True)
    else:
        print("  无（所有氨基酸准确率 >= 50%）", flush=True)

    print("\n优秀氨基酸 (准确率 >= 70%):", flush=True)
    excellent = [(aa, acc, total) for aa, acc, total, _, _ in results if acc >= 0.7]
    for aa, acc, total in excellent:
        print(f"  {aa}: {acc:.2%} ({total:,} 样本)", flush=True)

    print(f"\n{'='*60}", flush=True)

    # 保存结果
    output = {
        'per_aa_accuracy': [{'aa': r[0], 'accuracy': r[1], 'samples': r[2],
                              'true_main_codon': r[3], 'pred_main_codon': r[4]} for r in results if r[2] > 0],
        'bottlenecks': [{'aa': r[0], 'accuracy': r[1], 'samples': r[2]} for r in bottleneck],
        'excellent': [{'aa': r[0], 'accuracy': r[1], 'samples': r[2]} for r in excellent],
    }

    output_path = WORK_DIR / 'per_aa_accuracy.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {output_path}", flush=True)
    print(f"{'='*60}\n", flush=True)

if __name__ == '__main__':
    main()
