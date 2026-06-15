#!/usr/bin/env python3
"""
E. coli 逆翻译模型 - 多菌株真实数据训练 (10,110 条) - 修复版
数据来源: ecoli_real_cds_combined.json
菌株: MG1655 + BW2952 + DH1 + DH10B
架构：BiLSTM(2层,双向,h=128) + MultiheadAttention(4头) + FC
"""
import os
os.environ.setdefault('USERNAME', 'user')
os.environ.setdefault('HOME', 'C:/Users/lingongqi')
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import json, itertools, random, math, time, sys, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path
from collections import Counter

WORK_DIR = Path(__file__).parent

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==================== 配置 ====================
MAX_LEN    = 500
HIDDEN_DIM = 128
EPOCHS     = 50
BATCH_SIZE = 32  # 减小 batch size 以避免内存问题
SEED       = 42
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60, flush=True)
print("E. coli 逆翻译 - 多菌株真实 CDS 训练 (10,110 条) - 修复版", flush=True)
print(f"MAX_LEN={MAX_LEN}, EPOCHS={EPOCHS}, device={DEVICE}", flush=True)
print("=" * 60, flush=True)

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

ALL_CODONS = [''.join(c) for c in itertools.product('ATGC', repeat=3)]
CODON_TO_IDX = {c: i for i, c in enumerate(ALL_CODONS)}
AA_TO_IDX = {aa: i for i, aa in enumerate('ARNDCQEGHILKFPSWYVM')}
HYDROPHOBIC = {'A','V','I','L','M','F','W','Y'}
CONSERVED   = {'C','G','P','W'}

# ==================== 加载数据 ====================
def load_data():
    json_path = WORK_DIR / 'ecoli_real_cds_combined.json'
    print(f"\n[1/6] 加载多菌株数据: {json_path}", flush=True)
    
    try:
        with open(json_path, encoding='utf-8') as f:
            raw = json.load(f)
        
        data = [r for r in raw if 50 <= r['aa_len'] <= MAX_LEN]
        print(f"  加载 {len(data)} 条（过滤后，原 {len(raw)} 条）", flush=True)
        
        lens = [r['aa_len'] for r in data]
        print(f"  长度: avg={np.mean(lens):.0f}, med={np.median(lens):.0f}, {min(lens)}-{max(lens)}", flush=True)
        
        # 菌株分布
        sc = Counter(r.get('strain', 'unknown') for r in data)
        for s, c in sc.most_common():
            print(f"    {s}: {c}", flush=True)
        
        return data
    except Exception as e:
        print(f"  ✗ 加载数据失败: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

# ==================== 特征提取 ====================
def extract_features(data):
    print(f"\n[2/6] 提取特征（52维）...", flush=True)
    X = np.zeros((len(data), MAX_LEN, 52), dtype=np.float32)
    
    for idx, item in enumerate(tqdm(data, desc="特征提取")):
        protein = item['protein']
        prot_len = len(protein)
        
        for j, aa in enumerate(protein[:MAX_LEN]):
            feat = np.zeros(52, dtype=np.float32)
            ai = AA_TO_IDX.get(aa, 0)
            feat[ai] = 1.0
            
            # 二级结构 (3)
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
    
    print(f"  X shape: {X.shape}", flush=True)
    return X

def extract_labels(data):
    print(f"\n  提取标签...", flush=True)
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

# ==================== 训练 ====================
def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    best_val_acc, history = 0.0, []
    start = time.time()
    
    for epoch in range(1, epochs+1):
        print(f"\n  [Epoch {epoch}/{epochs}] 开始...", flush=True)
        model.train()
        total_loss, n = 0.0, 0
        
        try:
            for batch_idx, (Xb, yb) in enumerate(train_loader):
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(Xb).view(-1, 64), yb.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item(); n += 1
                
                if batch_idx % 50 == 0:
                    print(f"    Batch {batch_idx}: loss={loss.item():.4f}", flush=True)
            
            # 验证
            print(f"  [Epoch {epoch}] 验证中...", flush=True)
            model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    out = model(Xb.to(device)).argmax(-1).cpu().numpy()
                    for i in range(len(out)):
                        mask = yb[i].numpy() != 0
                        val_preds.extend(out[i][mask]); val_true.extend(yb[i].numpy()[mask])
            val_acc = accuracy_score(val_true, val_preds) if val_true else 0.0
            scheduler.step(1 - val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), WORK_DIR / 'best_combined_model.pth')
                print(f"  [Epoch {epoch}] ★ 新的最佳验证准确率: {val_acc:.4f}", flush=True)
            
            history.append({'epoch': epoch, 'loss': total_loss/n, 'val_acc': val_acc})
            elapsed = time.time() - start
            
            print(f"  Epoch {epoch:3d}/{epochs} | Loss={total_loss/n:.4f} | Val Acc={val_acc:.4f} | Best={best_val_acc:.4f} | {elapsed:.0f}s", flush=True)
        
        except Exception as e:
            print(f"  [Epoch {epoch}] ✗ 错误: {e}", flush=True)
            traceback.print_exc()
            break
    
    return history, best_val_acc, time.time() - start

# ==================== 主函数 ====================
def main():
    try:
        # 1. 数据
        data = load_data()
        total = len(data)
        
        # 2. 特征 & 标签
        X = extract_features(data)
        y = extract_labels(data)
        
        # 3. 划分: 75%训练 / 12.5%验证 / 12.5%测试
        print(f"\n[3/6] 划分数据集...", flush=True)
        indices = list(range(total))
        train_val_idx, test_idx = train_test_split(indices, test_size=1264, random_state=42)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=1264, random_state=42)
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]
        
        print(f"  训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}", flush=True)
        
        # 4. DataLoader
        print(f"\n[4/6] 创建 DataLoader (batch_size={BATCH_SIZE})...", flush=True)
        train_loader = DataLoader(CodonDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(CodonDataset(X_val, y_val),     batch_size=BATCH_SIZE, shuffle=False)
        test_loader  = DataLoader(CodonDataset(X_test, y_test),   batch_size=BATCH_SIZE, shuffle=False)
        print(f"  训练 batches: {len(train_loader)}", flush=True)
        
        # 5. 模型
        print(f"\n[5/6] BiLSTM + Attention 模型...", flush=True)
        model = BackTranslationModel()
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}", flush=True)
        
        # 6. 训练
        print(f"\n[6/6] 训练 {EPOCHS} epochs...", flush=True)
        history, best_val_acc, elapsed = train_model(model, train_loader, val_loader, DEVICE, epochs=EPOCHS)
        
        # 7. 测试
        print(f"\n  加载最佳模型...", flush=True)
        model.load_state_dict(torch.load(WORK_DIR / 'best_combined_model.pth', map_location=DEVICE))
        model.to(DEVICE); model.eval()
        
        print(f"  测试中...", flush=True)
        all_preds, all_true = [], []
        with torch.no_grad():
            for Xb, yb in test_loader:
                out = model(Xb.to(DEVICE)).argmax(-1).cpu().numpy()
                for i in range(len(out)):
                    mask = yb[i].numpy() != 0
                    all_preds.extend(out[i][mask]); all_true.extend(yb[i].numpy()[mask])
        test_acc = accuracy_score(all_true, all_preds)
        
        # 8. 结果
        print(f"\n{'='*60}", flush=True)
        print(f"多菌株 BiLSTM 测试准确率: {test_acc*100:.2f}%", flush=True)
        print(f"基线: 合成 74.14% | 单菌株 MG1655 55.22%", flush=True)
        print(f"vs 合成: {test_acc-0.7414:+.2%} | vs 单菌株: {test_acc-0.5522:+.2%}", flush=True)
        print(f"训练时间: {elapsed/60:.1f}min", flush=True)
        print(f"{'='*60}", flush=True)
        
        results = {
            'data_type': 'real_ncbi_multistrain',
            'strains': ['MG1655','BW2952','DH1','DH10B'],
            'total_records': total,
            'train_size': len(X_train), 'val_size': len(X_val), 'test_size': len(X_test),
            'max_len': MAX_LEN, 'epochs': EPOCHS,
            'best_val_acc': float(best_val_acc), 'test_acc': float(test_acc),
            'baseline_synthetic_74': 0.7414, 'baseline_single_strain_55': 0.5522,
            'vs_synthetic': float(test_acc - 0.7414),
            'vs_single_strain': float(test_acc - 0.5522),
            'training_time_s': float(elapsed),
            'history': history,
        }
        results_path = WORK_DIR / 'combined_model_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果保存: {results_path}", flush=True)
        
    except Exception as e:
        print(f"\n✗ 主函数错误: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
