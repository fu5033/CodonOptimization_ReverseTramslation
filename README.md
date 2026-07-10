================================================================================
  E. coli Back-Translation Model — Real Multi-Strain CDS Training Project
  fuhg2026-6
================================================================================

一、Project Overview
------------
```text
  Task: Amino acid sequence $\rightarrow$ Codon sequence "Back-Translation"  
  Organism: Escherichia coli (E. coli)  
  Data: NCBI RefSeq real CDS sequences, combined across 4 strains  
  Model: BiLSTM (2-layer bidirectional) + MultiheadAttention (4 heads) + Fully Connected Classification Laye  
  Best Test Accuracy: 60.80% (The optimal result achieved on real-world data)  
```
二、File List
------------
```text
  [Core Files]  
  train_realdata_combined_fixed.py   Training script (Main program)  
  best_combined_model.pth            Trained model weights (~3.8 MB, PyTorch .pth format)  
  combined_model_results.json        Training results (Loss/val_acc per epoch + final test accuracy)  
  ecoli_real_cds_combined.json       Multi-strain combined dataset (10,110 real CDS sequences)

  [Helper Scripts]  
  per_aa_analysis.py                 Per-Amino Acid (Per-AA) accuracy analysis script  
  multi_organism_codon_usage.py      Codon usage frequency database across multiple species  

  [Readme Text]  
  README.txt                         This file
```
三、Dataset Details
--------------
```text
  Source: NCBI RefSeq，E. coli CDS Coding Regions  
  Strains: MG1655 + BW2952 + DH1 + DH10B  
  Total Sequences: 10,110 sequences（Filtered:50 <= amino acid length <= 500）  
  Dataset Split (Random Seed = 42):  
    Training Set: 7,582 sequences  (75%)  
    Validation Set: 1,264 sequences  (12.5%)  
    Test Set: 1,264 sequences  (12.5%)  
  Data Format (JSON Array, each record contains):  
    - strain:  Strain name（e.g., "MG1655"）  
    - protein: Amino acid sequence（e.g., "MKKI..."）  
    - codons:  Codon sequence（e.g., ["ATG","AAA","AAA","ATT",...]）  
    - aa_len:  Amino acid sequence length  
    - dna_len: Codon sequence length（= aa_len * 3）  
```
四、Model Architecture
------------
```text
BackTranslationModel:
┌───────────────────────────────────────────────────┐
│  Input: 52-dimensional feature vector             │
│    [0-19]  AA one-hot (20-dim)                    │
│   [20-22]  Secondary structure heuristic (3-dim)  │
│     [23]   Hydrophobicity (1-dim)                 │
│     [24]   Conservation (1-dim)                   │
│     [25]   Relative position (1-dim)              │
│     [26]   sin positional encoding (1-dim)        │
│   [27-31]  Codon frequency prior (5-dim)          │
├───────────────────────────────────────────────────┤
│  BiLSTM (2layers, h=128, bidirectional, dr=0.3)   │
├───────────────────────────────────────────────────┤
│  LayerNorm (hidden*2 = 256)                       │
├───────────────────────────────────────────────────┤
│  MultiheadAttention (4 heads, dr=0.3)             │
├───────────────────────────────────────────────────┤
│  Residual Connection: LSTM_out + Attn_out         │
├───────────────────────────────────────────────────┤
│  FC Layers (256 -> 128 -> 64 output classes)      │
│  Includes ReLU + Dropout                          │
└───────────────────────────────────────────────────┘
```
```text
  Output Classes: 64 codons (including stop codons, contains T)
  Parameter size: Approx 1,000,000+
  Maximum Input Sequence Length: MAX_LEN = 500
```
五、Training Configuration
------------
```text
  Optimizer：Adam (lr=1e-3, weight_decay=1e-5)  
  Scheduler：ReduceLROnPlateau (patience=5, factor=0.5)  
  Loss Function：CrossEntropyLoss (ignore_index=0, ignores stop codons)  
  Gradient Clipping：max_norm=1.0  
  Batch Size：32  
  Training Epochs：50 epochs  
  Device：CPU（PyTorch）  
  Total Training Time：Approx 25.4 hours（91,505 seconds）  
```
六、Training Results
------------
```text
  ┌──────────┬───────────────┬──────────┐
  │ Metric   │ Validation Set│ Test Set │
  ├──────────┼───────────────┼──────────┤
  │ Accuracy │   60.40%      │  60.80%  │
  └──────────┴───────────────┴──────────┘
```
```text
  Baseline Comparison:  
    - Synthetic Data（Same architecture, 4000 sequences）：74.14%  
    - Single-Strain MG1655 Real Data：55.22%  
    - Multi-Strain Combined (This Project)：60.80% ← Current Best  

  Training Curve (Key Milestones):  
    Epoch  1:  Val Acc = 56.43%  
    Epoch 10:  Val Acc = 59.06%  
    Epoch 27:  Val Acc = 60.12%  (First time breaking 60%)  
    Epoch 45:  Val Acc = 60.40%  (Peak performance, best model saved)  
    Epoch 50:  Val Acc = 60.07%  

  Per-AA Analysis（(Key Bottlenecks）：
    - S (Serine, 6 codons)：Lowest accuracy，~38.6%
    - R (Arginine, 6 codons)：Second lowest，~50.4%
    - M (Methionine, single codon)：~100%
    - K (Lysine, 2 codons)：~100%
    - W (Tryptophan, single codon)：~100%
    Note: Run per_aa_analysis.py to get the full per-amino acid statistics.
```
七、Model Usage (Inference Example)
------------------------

  1. Environment Requirements:
```text
     - Python 3.8+
     - PyTorch >= 1.9
     - NumPy, scikit-learn
```
  2. Loading the Model for Inference:
  ```python
  import torch
  import numpy as np
  from train_realdata_combined_fixed import BackTranslationModel, extract_features

  # Load the model
  model = BackTranslationModel(input_dim=52, hidden_dim=128, num_codons=64)
  model.load_state_dict(torch.load('best_combined_model.pth', map_location='cpu'))
  model.eval()

  # Prepare inputs（list of amino acid sequences + corresponding DNA lengths）
  # data = [{'protein': 'MKK...', 'aa_len': 150}, ...]
  # X = extract_features(data)  # shape: [N, 500, 52]
  # X_tensor = torch.FloatTensor(X)

  # Prediction
  # with torch.no_grad():
  #     out = model(X_tensor)  # shape: [N, 500, 64]
  #     pred_indices = out.argmax(-1)  # Predicted codon index for each position
  ```

  3. run Per-AA Analysis：
```text
     $ python per_aa_analysis.py
     （Requires ecoli_real_cds_combined.json and best_combined_model.pth to be in the same directory）
```
八、Code Description
------------
```text
  train_realdata_combined_fixed.py
    The complete training pipeline, including the following steps:
    1. Load JSON data, filter by length (50-500 AA).
    2. Extract 52-dimensional features (AA one-hot + structure + physicochemical properties + positional encoding + codon frequency).
    3. Extract labels (codon indices)
    4. Split into training/validation/test sets at a 75 / 12.5 / 12.5 ratio
    5. Construct the BiLSTM + Attention model
    6. Train for 50 epochs, validate each round, and save the best model
    7. Evaluate finally on the test set

  multi_organism_codon_usage.py
    Contains codon usage frequency databases and helper query functions for 7 species (ecoli, yeast, human, mouse, fly, arabidopsis, rice).

  per_aa_analysis.py
    Load the best model and calculate the accuracy rate of codon prediction by amino acid groups on the test set，
    Identify bottleneck amino acids (with accuracy < 50%) and excellent amino acids (with accuracy >= 70%).
```
九、Known Limitations
------------
```text
  1. The feature dimension is 52, which only retains the top 5 codons for the codon frequency prior. For amino acids with 6 codons (e.g., S, R, L), the 6th codon information is truncated.
  2. Training was completed entirely on a CPU. Each epoch took around 30 minutes, totaling roughly 25 hours for 50 epochs.
  3. Real-world codon choice is influenced by multiple biological factors (tRNA abundance, mRNA stability, translation efficiency, etc.). The model only models frequency distributions, so its accuracy (60.80%) is noticeably lower than on synthetic data (74.14%).
  4. The dataset is derived from 4 E. coli laboratory strains and may not generalize well to wild-type strains or other bacterial species.
```
十、Future Optimization Directions (Already Attempted)
--------------------------
```text
  Version Comparison:
  ┌──────────────┬───────────────────────────────────────┬──────────────┐
  │ Version      │ Enhancements                          │ Test Accuracy│
  ├──────────────┼───────────────────────────────────────┼──────────────┤
  │ combined     │ 52-dim + BiLSTM + MAX_LEN=500         │   60.80%     │
  │ v3_fast      │ 62-dim + InputProj + MAX_LEN=200      │   55.43%     │
  │ v4_max500    │ 62-dim + pack_padded + LEN=500        │   57.75%*    │
  │ v4_resume    │ Same as above, resumed training to E10│   58.98%*    │
  └──────────────┴───────────────────────────────────────┴──────────────┘
```
```text
  * The training process was unstable on the CPU and did not complete all 50 epochs.

  Recommendation: Running on a GPU can drastically accelerate training and allow for experimenting with larger models.
```
================================================================================
  Generation Date：2026-06-12
  Author：fuhg
================================================================================
