"""
多物种密码子使用频率数据库
包含：大肠杆菌、酵母、人类、小鼠等常见生物
数据来源：Kazusa DNA Research Institute 和文献报道
"""

# 通用遗传密码（所有生物共享的氨基酸-密码子映射）
AA_TO_CODONS = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'N': ['AAT', 'AAC'],
    'D': ['GAT', 'GAC'],
    'C': ['TGT', 'TGC'],
    'Q': ['CAA', 'CAG'],
    'E': ['GAA', 'GAG'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],
    'H': ['CAT', 'CAC'],
    'I': ['ATT', 'ATC', 'ATA'],
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'K': ['AAA', 'AAG'],
    'M': ['ATG'],
    'F': ['TTT', 'TTC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    'W': ['TGG'],
    'Y': ['TAT', 'TAC'],
    'V': ['GTT', 'GTC', 'GTA', 'GTG']
}

# ==================== 大肠杆菌 (Escherichia coli) ====================
ECOLI_CODON_FREQ = {
    'GCT': 0.18, 'GCC': 0.27, 'GCA': 0.23, 'GCG': 0.32,
    'CGT': 0.22, 'CGC': 0.28, 'CGA': 0.05, 'CGG': 0.05, 'AGA': 0.04, 'AGG': 0.04,
    'AAT': 0.49, 'AAC': 0.51,
    'GAT': 0.63, 'GAC': 0.37,
    'TGT': 0.46, 'TGC': 0.54,
    'CAA': 0.34, 'CAG': 0.66,
    'GAA': 0.68, 'GAG': 0.32,
    'GGT': 0.25, 'GGC': 0.25, 'GGA': 0.15, 'GGG': 0.35,
    'CAT': 0.47, 'CAC': 0.53,
    'ATT': 0.51, 'ATC': 0.25, 'ATA': 0.24,
    'TTA': 0.14, 'TTG': 0.13, 'CTT': 0.12, 'CTC': 0.11, 'CTA': 0.04, 'CTG': 0.52,
    'AAA': 0.76, 'AAG': 0.24,
    'TTT': 0.58, 'TTC': 0.42,
    'CCT': 0.18, 'CCC': 0.12, 'CCA': 0.20, 'CCG': 0.50,
    'TCT': 0.12, 'TCC': 0.15, 'TCA': 0.14, 'TCG': 0.15, 'AGT': 0.09, 'AGC': 0.28,
    'ACT': 0.18, 'ACC': 0.40, 'ACA': 0.12, 'ACG': 0.30,
    'TGG': 1.00,
    'TAT': 0.59, 'TAC': 0.41,
    'GTT': 0.18, 'GTC': 0.15, 'GTA': 0.11, 'GTG': 0.56,
    'ATG': 1.00
}

# ==================== 酿酒酵母 (Saccharomyces cerevisiae) ====================
YEAST_CODON_FREQ = {
    'GCT': 0.38, 'GCC': 0.22, 'GCA': 0.32, 'GCG': 0.08,
    'CGT': 0.15, 'CGC': 0.06, 'CGA': 0.06, 'CGG': 0.06, 'AGA': 0.48, 'AGG': 0.19,
    'AAT': 0.59, 'AAC': 0.41,
    'GAT': 0.65, 'GAC': 0.35,
    'TGT': 0.65, 'TGC': 0.35,
    'CAA': 0.69, 'CAG': 0.31,
    'GAA': 0.70, 'GAG': 0.30,
    'GGT': 0.24, 'GGC': 0.20, 'GGA': 0.29, 'GGG': 0.27,
    'CAT': 0.67, 'CAC': 0.33,
    'ATT': 0.63, 'ATC': 0.29, 'ATA': 0.08,
    'TTA': 0.29, 'TTG': 0.28, 'CTT': 0.13, 'CTC': 0.06, 'CTA': 0.14, 'CTG': 0.10,
    'AAA': 0.59, 'AAG': 0.41,
    'TTT': 0.67, 'TTC': 0.33,
    'CCT': 0.36, 'CCC': 0.12, 'CCA': 0.33, 'CCG': 0.19,
    'TCT': 0.32, 'TCC': 0.16, 'TCA': 0.30, 'TCG': 0.08, 'AGT': 0.10, 'AGC': 0.04,
    'ACT': 0.40, 'ACC': 0.28, 'ACA': 0.26, 'ACG': 0.06,
    'TGG': 1.00,
    'TAT': 0.65, 'TAC': 0.35,
    'GTT': 0.41, 'GTC': 0.21, 'GTA': 0.20, 'GTG': 0.18,
    'ATG': 1.00
}

# ==================== 人类 (Homo sapiens) ====================
HUMAN_CODON_FREQ = {
    'GCT': 0.27, 'GCC': 0.40, 'GCA': 0.23, 'GCG': 0.10,
    'CGT': 0.08, 'CGC': 0.19, 'CGA': 0.11, 'CGG': 0.21, 'AGA': 0.21, 'AGG': 0.20,
    'AAT': 0.47, 'AAC': 0.53,
    'GAT': 0.46, 'GAC': 0.54,
    'TGT': 0.46, 'TGC': 0.54,
    'CAA': 0.27, 'CAG': 0.73,
    'GAA': 0.42, 'GAG': 0.58,
    'GGT': 0.24, 'GGC': 0.34, 'GGA': 0.16, 'GGG': 0.26,
    'CAT': 0.42, 'CAC': 0.58,
    'ATT': 0.36, 'ATC': 0.53, 'ATA': 0.11,
    'TTA': 0.07, 'TTG': 0.13, 'CTT': 0.13, 'CTC': 0.20, 'CTA': 0.07, 'CTG': 0.40,
    'AAA': 0.43, 'AAG': 0.57,
    'TTT': 0.45, 'TTC': 0.55,
    'CCT': 0.28, 'CCC': 0.33, 'CCA': 0.27, 'CCG': 0.12,
    'TCT': 0.18, 'TCC': 0.22, 'TCA': 0.15, 'TCG': 0.05, 'AGT': 0.15, 'AGC': 0.25,
    'ACT': 0.24, 'ACC': 0.36, 'ACA': 0.28, 'ACG': 0.12,
    'TGG': 1.00,
    'TAT': 0.44, 'TAC': 0.56,
    'GTT': 0.18, 'GTC': 0.24, 'GTA': 0.11, 'GTG': 0.47,
    'ATG': 1.00
}

# ==================== 小鼠 (Mus musculus) ====================
MOUSE_CODON_FREQ = {
    'GCT': 0.26, 'GCC': 0.39, 'GCA': 0.23, 'GCG': 0.12,
    'CGT': 0.07, 'CGC': 0.16, 'CGA': 0.07, 'CGG': 0.14, 'AGA': 0.27, 'AGG': 0.29,
    'AAT': 0.43, 'AAC': 0.57,
    'GAT': 0.44, 'GAC': 0.56,
    'TGT': 0.48, 'TGC': 0.52,
    'CAA': 0.25, 'CAG': 0.75,
    'GAA': 0.39, 'GAG': 0.61,
    'GGT': 0.22, 'GGC': 0.32, 'GGA': 0.14, 'GGG': 0.32,
    'CAT': 0.39, 'CAC': 0.61,
    'ATT': 0.35, 'ATC': 0.50, 'ATA': 0.15,
    'TTA': 0.05, 'TTG': 0.11, 'CTT': 0.11, 'CTC': 0.16, 'CTA': 0.05, 'CTG': 0.52,
    'AAA': 0.38, 'AAG': 0.62,
    'TTT': 0.43, 'TTC': 0.57,
    'CCT': 0.27, 'CCC': 0.27, 'CCA': 0.26, 'CCG': 0.20,
    'TCT': 0.17, 'TCC': 0.19, 'TCA': 0.14, 'TCG': 0.04, 'AGT': 0.14, 'AGC': 0.32,
    'ACT': 0.21, 'ACC': 0.33, 'ACA': 0.28, 'ACG': 0.18,
    'TGG': 1.00,
    'TAT': 0.42, 'TAC': 0.58,
    'GTT': 0.16, 'GTC': 0.22, 'GTA': 0.10, 'GTG': 0.52,
    'ATG': 1.00
}

# ==================== 果蝇 (Drosophila melanogaster) ====================
FLY_CODON_FREQ = {
    'GCT': 0.38, 'GCC': 0.29, 'GCA': 0.25, 'GCG': 0.08,
    'CGT': 0.20, 'CGC': 0.17, 'CGA': 0.07, 'CGG': 0.08, 'AGA': 0.18, 'AGG': 0.30,
    'AAT': 0.56, 'AAC': 0.44,
    'GAT': 0.63, 'GAC': 0.37,
    'TGT': 0.54, 'TGC': 0.46,
    'CAA': 0.44, 'CAG': 0.56,
    'GAA': 0.59, 'GAG': 0.41,
    'GGT': 0.32, 'GGC': 0.23, 'GGA': 0.22, 'GGG': 0.23,
    'CAT': 0.52, 'CAC': 0.48,
    'ATT': 0.50, 'ATC': 0.38, 'ATA': 0.12,
    'TTA': 0.12, 'TTG': 0.18, 'CTT': 0.14, 'CTC': 0.10, 'CTA': 0.09, 'CTG': 0.37,
    'AAA': 0.54, 'AAG': 0.46,
    'TTT': 0.58, 'TTC': 0.42,
    'CCT': 0.32, 'CCC': 0.19, 'CCA': 0.30, 'CCG': 0.19,
    'TCT': 0.25, 'TCC': 0.19, 'TCA': 0.22, 'TCG': 0.07, 'AGT': 0.13, 'AGC': 0.14,
    'ACT': 0.29, 'ACC': 0.26, 'ACA': 0.24, 'ACG': 0.21,
    'TGG': 1.00,
    'TAT': 0.55, 'TAC': 0.45,
    'GTT': 0.29, 'GTC': 0.22, 'GTA': 0.17, 'GTG': 0.32,
    'ATG': 1.00
}

# ==================== 水稻 (Oryza sativa) ====================
# 数据来源: Kazusa Codon Usage Database (O. sativa japonica, high-expression genes)
RICE_CODON_FREQ = {
    # Alanine (GC*)
    'GCT': 0.17, 'GCC': 0.32, 'GCA': 0.19, 'GCG': 0.32,
    # Arginine (CG* + AG*)
    'CGT': 0.12, 'CGC': 0.26, 'CGA': 0.08, 'CGG': 0.22, 'AGA': 0.16, 'AGG': 0.16,
    # Asparagine (AA*)
    'AAT': 0.32, 'AAC': 0.68,
    # Aspartate (GA*)
    'GAT': 0.37, 'GAC': 0.63,
    # Cysteine (TG*)
    'TGT': 0.39, 'TGC': 0.61,
    # Glutamine (CA*)
    'CAA': 0.24, 'CAG': 0.76,
    # Glutamate (GA*)
    'GAA': 0.34, 'GAG': 0.66,
    # Glycine (GG*)
    'GGT': 0.17, 'GGC': 0.39, 'GGA': 0.18, 'GGG': 0.26,
    # Histidine (CA*)
    'CAT': 0.34, 'CAC': 0.66,
    # Isoleucine (AT*)
    'ATT': 0.26, 'ATC': 0.58, 'ATA': 0.16,
    # Leucine (TT* + CT*)
    'TTA': 0.04, 'TTG': 0.09, 'CTT': 0.12, 'CTC': 0.22, 'CTA': 0.06, 'CTG': 0.47,
    # Lysine (AA*)
    'AAA': 0.33, 'AAG': 0.67,
    # Methionine
    'ATG': 1.00,
    # Phenylalanine (TT*)
    'TTT': 0.31, 'TTC': 0.69,
    # Proline (CC*)
    'CCT': 0.13, 'CCC': 0.22, 'CCA': 0.18, 'CCG': 0.47,
    # Serine (TC* + AG*)
    'TCT': 0.10, 'TCC': 0.22, 'TCA': 0.11, 'TCG': 0.25, 'AGT': 0.10, 'AGC': 0.22,
    # Threonine (AC*)
    'ACT': 0.15, 'ACC': 0.37, 'ACA': 0.16, 'ACG': 0.32,
    # Tryptophan
    'TGG': 1.00,
    # Tyrosine (TA*)
    'TAT': 0.30, 'TAC': 0.70,
    # Valine (GT*)
    'GTT': 0.14, 'GTC': 0.26, 'GTA': 0.11, 'GTG': 0.49,
}

# ==================== 拟南芥 (Arabidopsis thaliana) ====================
ARABIDOPSIS_CODON_FREQ = {
    'GCT': 0.22, 'GCC': 0.31, 'GCA': 0.34, 'GCG': 0.13,
    'CGT': 0.11, 'CGC': 0.15, 'CGA': 0.14, 'CGG': 0.13, 'AGA': 0.14, 'AGG': 0.13,
    'AAT': 0.48, 'AAC': 0.52,
    'GAT': 0.53, 'GAC': 0.47,
    'TGT': 0.42, 'TGC': 0.58,
    'CAA': 0.38, 'CAG': 0.62,
    'GAA': 0.45, 'GAG': 0.55,
    'GGT': 0.21, 'GGC': 0.26, 'GGA': 0.29, 'GGG': 0.24,
    'CAT': 0.44, 'CAC': 0.56,
    'ATT': 0.32, 'ATC': 0.28, 'ATA': 0.40,
    'TTA': 0.07, 'TTG': 0.10, 'CTT': 0.12, 'CTC': 0.10, 'CTA': 0.06, 'CTG': 0.55,
    'AAA': 0.41, 'AAG': 0.59,
    'TTT': 0.50, 'TTC': 0.50,
    'CCT': 0.19, 'CCC': 0.16, 'CCA': 0.35, 'CCG': 0.30,
    'TCT': 0.15, 'TCC': 0.19, 'TCA': 0.20, 'TCG': 0.15, 'AGT': 0.16, 'AGC': 0.15,
    'ACT': 0.19, 'ACC': 0.30, 'ACA': 0.24, 'ACG': 0.27,
    'TGG': 1.00,
    'TAT': 0.48, 'TAC': 0.52,
    'GTT': 0.19, 'GTC': 0.15, 'GTA': 0.21, 'GTG': 0.45,
    'ATG': 1.00
}

# ==================== 物种元数据 ====================
ORGANISM_METADATA = {
    'ecoli': {
        'name': 'Escherichia coli',
        'type': 'bacteria',
        'codon_freq': ECOLI_CODON_FREQ,
        'gc_content': 0.51,
        'description': '大肠杆菌 - 原核模式生物'
    },
    'yeast': {
        'name': 'Saccharomyces cerevisiae',
        'type': 'fungi',
        'codon_freq': YEAST_CODON_FREQ,
        'gc_content': 0.38,
        'description': '酿酒酵母 - 真核模式生物'
    },
    'human': {
        'name': 'Homo sapiens',
        'type': 'mammal',
        'codon_freq': HUMAN_CODON_FREQ,
        'gc_content': 0.52,
        'description': '人类'
    },
    'mouse': {
        'name': 'Mus musculus',
        'type': 'mammal',
        'codon_freq': MOUSE_CODON_FREQ,
        'gc_content': 0.51,
        'description': '小鼠'
    },
    'fly': {
        'name': 'Drosophila melanogaster',
        'type': 'insect',
        'codon_freq': FLY_CODON_FREQ,
        'gc_content': 0.52,
        'description': '黑腹果蝇'
    },
    'arabidopsis': {
        'name': 'Arabidopsis thaliana',
        'type': 'plant',
        'codon_freq': ARABIDOPSIS_CODON_FREQ,
        'gc_content': 0.46,
        'description': '拟南芥 - 植物模式生物'
    },
    'rice': {
        'name': 'Oryza sativa',
        'type': 'plant',
        'codon_freq': RICE_CODON_FREQ,
        'gc_content': 0.55,
        'description': '水稻 - 单子叶植物模式生物，重要粮食作物'
    }
}

# ==================== 辅助函数 ====================
def get_organism_list():
    """返回支持的物种列表"""
    return list(ORGANISM_METADATA.keys())

def get_codon_freq(organism):
    """获取指定物种的密码子频率表"""
    if organism in ORGANISM_METADATA:
        return ORGANISM_METADATA[organism]['codon_freq']
    else:
        raise ValueError(f"不支持的物种: {organism}. 支持的物种: {get_organism_list()}")

def get_all_codon_freq():
    """获取所有物种的密码子频率表"""
    return {org: meta['codon_freq'] for org, meta in ORGANISM_METADATA.items()}

def print_organism_info(organism):
    """打印物种信息"""
    if organism in ORGANISM_METADATA:
        meta = ORGANISM_METADATA[organism]
        print(f"\n物种: {organism}")
        print(f"  学名: {meta['name']}")
        print(f"  类型: {meta['type']}")
        print(f"  GC含量: {meta['gc_content']:.2%}")
        print(f"  描述: {meta['description']}")
    else:
        print(f"未知物种: {organism}")

def print_all_organisms():
    """打印所有支持的物种"""
    print("\n支持的物种列表:")
    print("=" * 60)
    for org in get_organism_list():
        print_organism_info(org)

if __name__ == '__main__':
    # 测试代码
    print_all_organisms()
    
    print("\n\n示例：大肠杆菌的Ala密码子使用频率:")
    ecoli_freq = get_codon_freq('ecoli')
    for codon in ['GCT', 'GCC', 'GCA', 'GCG']:
        print(f"  {codon}: {ecoli_freq[codon]:.2%}")
