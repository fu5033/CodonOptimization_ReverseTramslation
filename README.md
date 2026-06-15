================================================================================
  大肠杆菌（E. coli）逆翻译模型 —— 真实多菌株 CDS 训练项目
  fuhg2026-6
================================================================================

一、项目概述
------------
  任务：氨基酸序列 → 密码子序列的"反向翻译"（Back-Translation）
  物种：大肠杆菌（Escherichia coli）
  数据：NCBI RefSeq 真实 CDS 序列，4 个菌株合并
  模型：BiLSTM（2层双向）+ MultiheadAttention（4头）+ 全连接分类层
  最佳测试准确率：60.80%（真实数据上的最优结果）

二、文件清单
------------
  [核心文件]
  train_realdata_combined_fixed.py  训练脚本（主程序）
  best_combined_model.pth           训练好的模型权重（~3.8 MB，PyTorch .pth 格式）
  combined_model_results.json       训练结果（每轮 loss/val_acc + 最终测试准确率）
  ecoli_real_cds_combined.json      多菌株合并数据集（10,110 条真实 CDS）

  [辅助脚本]
  per_aa_analysis.py                逐氨基酸（Per-AA）准确率分析脚本
  multi_organism_codon_usage.py     多物种密码子使用频率数据库

  [说明文档]
  README.txt                        本文件

三、数据集详情
--------------
  来源：NCBI RefSeq，大肠杆菌 CDS 编码区
  菌株：MG1655 + BW2952 + DH1 + DH10B
  总序列数：10,110 条（过滤：50 <= 氨基酸长度 <= 500）
  数据集划分（随机种子=42）：
    训练集：7,582 条  (75%)
    验证集：1,264 条  (12.5%)
    测试集：1,264 条  (12.5%)
  数据格式（JSON 数组，每条记录包含）：
    - strain:  菌株名称（如 "MG1655"）
    - protein: 氨基酸序列（如 "MKKI..."）
    - codons:  密码子序列（如 ["ATG","AAA","AAA","ATT",...]）
    - aa_len:  氨基酸序列长度
    - dna_len: 密码子序列长度（= aa_len * 3）

四、模型架构
------------
  BackTranslationModel:
    ┌─────────────────────────────────┐
    │  Input: 52维特征向量             │
    │    [0-19]  AA one-hot (20维)    │
    │   [20-22]  二级结构启发式 (3维)  │
    │     [23]   疏水性 (1维)          │
    │     [24]   保守性 (1维)          │
    │     [25]   相对位置 (1维)        │
    │     [26]   sin位置编码 (1维)     │
    │   [27-31]  密码子频率先验 (5维)  │
    ├─────────────────────────────────┤
    │  BiLSTM (2层, h=128, 双向, dr=0.3) │
    ├─────────────────────────────────┤
    │  LayerNorm (hidden*2 = 256)     │
    ├─────────────────────────────────┤
    │  MultiheadAttention (4头, dr=0.3) │
    ├─────────────────────────────────┤
    │  残差连接: LSTM_out + Attn_out  │
    ├─────────────────────────────────┤
    │  FC 层 (256→128→64类输出)       │
    │  包含 ReLU + Dropout            │
    └─────────────────────────────────┘

  输出类别：64个密码子（包含终止密码子，含T）
  参数量：约 1,000,000+
  输入序列最大长度：MAX_LEN = 500

五、训练配置
------------
  优化器：Adam (lr=1e-3, weight_decay=1e-5)
  调度器：ReduceLROnPlateau (patience=5, factor=0.5)
  损失函数：CrossEntropyLoss (ignore_index=0, 忽略终止密码子)
  梯度裁剪：max_norm=1.0
  Batch Size：32
  训练轮次：50 epochs
  设备：CPU（PyTorch）
  总训练时间：约 25.4 小时（91,505 秒）

六、训练结果
------------
  ┌────────┬─────────┬──────────┐
  │ 指标   │ 验证集  │  测试集  │
  ├────────┼─────────┼──────────┤
  │ 准确率 │ 60.40%  │ 60.80%   │
  └────────┴─────────┴──────────┘

  对比基线：
    - 合成数据（同架构，4000条）：74.14%
    - 单菌株 MG1655 真实数据：55.22%
    - 多菌株合并（本结果）：60.80% ← 当前最佳

  训练曲线（关键节点）：
    Epoch  1:  Val Acc = 56.43%
    Epoch 10:  Val Acc = 59.06%
    Epoch 27:  Val Acc = 60.12%  (首次突破 60%)
    Epoch 45:  Val Acc = 60.40%  (峰值，保存最佳模型)
    Epoch 50:  Val Acc = 60.07%

  Per-AA 分析（关键瓶颈）：
    - S (Serine, 6密码子)：准确率最低，~38.6%
    - R (Arginine, 6密码子)：次低，~50.4%
    - M (Methionine, 单密码子)：~100%
    - K (Lysine, 2密码子)：~100%
    - W (Tryptophan, 单密码子)：~100%
    注：运行 per_aa_analysis.py 可获得完整逐氨基酸统计

七、模型使用（推理示例）
------------------------
  1. 环境要求：
     - Python 3.8+
     - PyTorch >= 1.9
     - NumPy, scikit-learn

  2. 加载模型进行预测：
  ```python
  import torch
  import numpy as np
  from train_realdata_combined_fixed import BackTranslationModel, extract_features

  # 加载模型
  model = BackTranslationModel(input_dim=52, hidden_dim=128, num_codons=64)
  model.load_state_dict(torch.load('best_combined_model.pth', map_location='cpu'))
  model.eval()

  # 准备输入（氨基酸序列列表 + 对应DNA长度）
  # data = [{'protein': 'MKK...', 'aa_len': 150}, ...]
  # X = extract_features(data)  # shape: [N, 500, 52]
  # X_tensor = torch.FloatTensor(X)

  # 预测
  # with torch.no_grad():
  #     out = model(X_tensor)  # shape: [N, 500, 64]
  #     pred_indices = out.argmax(-1)  # 每个位置的预测密码子索引
  ```

  3. 运行 Per-AA 分析：
     $ python per_aa_analysis.py
     （需 ecoli_real_cds_combined.json 和 best_combined_model.pth 在同一目录）

八、代码说明
------------
  train_realdata_combined_fixed.py
    完整的训练流程，包含以下步骤：
    1. 加载 JSON 数据，按长度过滤（50-500 AA）
    2. 提取 52 维特征（AA one-hot + 结构 + 理化属性 + 位置编码 + 密码子频率）
    3. 提取标签（密码子索引）
    4. 按 75/12.5/12.5 比例划分训练/验证/测试集
    5. 构建 BiLSTM + Attention 模型
    6. 训练 50 epoch，每轮验证，保存最佳模型
    7. 最终在测试集上评估

  multi_organism_codon_usage.py
    包含 7 个物种（ecoli, yeast, human, mouse, fly, arabidopsis, rice）的
    密码子使用频率数据库及辅助查询函数。

  per_aa_analysis.py
    加载最佳模型，在测试集上按氨基酸分组统计密码子预测准确率，
    识别瓶颈氨基酸（准确率 < 50%）和优秀氨基酸（准确率 >= 70%）。

九、已知限制
------------
  1. 特征维度为 52 维，其中密码子频率先验仅保留前 5 个密码子，
    对 S/R/L 等有 6 个密码子的氨基酸，第 6 个密码子信息被截断。
  2. 训练在 CPU 上完成，每 epoch 约 30 分钟，50 epoch 共 25 小时。
  3. 真实数据的密码子选择受多种因素影响（tRNA 丰度、mRNA 稳定性、
     翻译效率等），模型仅建模频率分布，因此准确率（60.80%）低于
     合成数据（74.14%）。
  4. 数据集来自 4 个大肠杆菌实验室菌株，可能不适用于野生型或
     其他细菌物种。

十、后续优化方向（已尝试）
--------------------------
  版本对比：
  ┌──────────────┬────────────────────────────────┬──────────┐
  │ 版本         │ 改进                           │ 测试准确率│
  ├──────────────┼────────────────────────────────┼──────────┤
  │ combined     │ 52维 + BiLSTM + MAX_LEN=500    │ 60.80%   │
  │ v3_fast      │ 62维 + InputProj + MAX_LEN=200 │ 55.43%   │
  │ v4_max500    │ 62维 + pack_padded + LEN=500   │ 57.75%*  │
  │ v4_resume    │ 同上，续训至 E10               │ 58.98%*  │
  └──────────────┴────────────────────────────────┴──────────┘
  * 训练进程在 CPU 上不稳定，未完成全部 50 轮。

  建议：在 GPU 上运行可大幅加速并尝试更大模型。

================================================================================
  生成日期：2026-06-12
  作者：fuhg
  项目目录：C:\Users\lingongqi\WorkBuddy\2026-05-10-task-3
================================================================================
