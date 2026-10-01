import numpy as np
import pandas as pd
import itertools
from typing import List, Dict, Union

class SequenceFeatureExtractor:
    """DNA序列特征提取器"""
    
    def __init__(self, seq_length=33):
        """初始化特征提取器"""
        self.bases = ['A', 'T', 'G', 'C']
        self.valid_bases = set('ATGCN')
        self.seq_length = seq_length  # 添加序列长度配置
        self.feature_names = self._get_feature_names()
    
    def _get_feature_names(self) -> List[str]:
        """
        返回特征名称列表，包含所有可能的碱基特征
        
        Returns:
            features: 特征名称列表
        """
        features = []

        # 基础特征 (包含N)
        bases = ['A', 'T', 'G', 'C', 'N']
        features.extend(bases)

        # 3-mer特征
        kmers_3 = [''.join(p) for p in itertools.product(bases, repeat=3)]  # 125个特征
        features.extend(kmers_3)

        # 密码子特征
        codons = [''.join(p) for p in itertools.product(['A', 'T', 'G', 'C'], repeat=3)]  # 64个密码子
        n_codons = self.seq_length // 3  # 计算序列中包含的完整密码子数量
        for i in range(n_codons):
            for codon in codons:
                features.append(f'codon_pos_{i}_{codon}')

        # GC含量特征
        features.append('gc_content')

        # 序列复杂度特征
        features.append('sequence_complexity')

        return features

    def trim_sequence(self, seq, target_length):
        """Center-crop to exactly target_length, removing an odd extra base on the right."""
        if len(seq) <= target_length:
            return seq
        start = (len(seq) - target_length) // 2
        return seq[start:start + target_length]
    
    def _preprocess_sequence(self, sequence):
        """
        将DNA序列转换为特征向量
        
        Args:
            sequence: DNA序列
            
        Returns:
            feature_vector: 特征向量
        """
        try:
            feature_names = self.feature_names
            
            if pd.isna(sequence) or not isinstance(sequence, str):
                sequence = str(sequence)
            sequence = sequence.upper().replace('U', 'T')  # 统一为大写字母
            sequence = ''.join(base if base in self.valid_bases else 'N' for base in sequence)

            # 如果序列长度不等于目标长度，进行截取或填充
            if len(sequence) > self.seq_length:
                sequence = self.trim_sequence(sequence, self.seq_length)
            else:
                sequence = sequence[:self.seq_length].ljust(self.seq_length, 'N')

            # 初始化特征字典
            features = {
                'A': 0,
                'T': 0,
                'G': 0,
                'C': 0,
                'N': 0
            }
            kmer_features = {}

            # 碱基组成
            for base in ['A', 'T', 'G', 'C', 'N']:
                features[base] = sequence.count(base) / self.seq_length

            # 3-mer特征
            for kmer in [''.join(p) for p in itertools.product(['A', 'T', 'G', 'C', 'N'], repeat=3)]:
                kmer_count = 0
                for i in range(self.seq_length - 2):
                    if sequence[i:i+3] == kmer:
                        kmer_count += 1
                kmer_features[kmer] = kmer_count / max(1, self.seq_length - 2)

            # 密码子特征
            codon_features = {}
            codons = [''.join(p) for p in itertools.product(['A', 'T', 'G', 'C'], repeat=3)]  # 64个密码子
            n_codons = self.seq_length // 3  # 计算序列中包含的完整密码子数量
            for i in range(n_codons):
                pos_start = i * 3
                current_codon = sequence[pos_start:pos_start+3]
                for codon in codons:
                    codon_features[f'codon_pos_{i}_{codon}'] = 1 if current_codon == codon and 'N' not in current_codon else 0

            # GC含量
            valid_bases = [b for b in sequence if b != 'N']
            gc_content = (valid_bases.count('G') + valid_bases.count('C')) / len(valid_bases) if valid_bases else 0

            # 序列复杂度（Shannon熵）
            from collections import Counter
            valid_counts = Counter(valid_bases)
            total_valid = sum(valid_counts.values())
            entropy = 0
            for cnt in valid_counts.values():
                p = cnt / total_valid
                entropy += -p * np.log2(p)
            entropy /= np.log2(4)  # 归一化到0-1

            # 合并所有特征
            all_features = {**features, **kmer_features, **codon_features}
            all_features['gc_content'] = gc_content
            all_features['sequence_complexity'] = entropy

            # 确保特征顺序一致
            feature_vector = [all_features.get(f, 0.0) for f in feature_names]

            return feature_vector
        except Exception as e:
            raise ValueError(f"特征提取失败: {str(e)}")
    
    def extract_features_batch(self, sequences: List[Union[str, float]]) -> np.ndarray:
        """
        批量提取特征
        
        Args:
            sequences: DNA序列列表
            
        Returns:
            np.ndarray: 特征矩阵
        """
        try:
            return np.array([self.extract_features(seq) for seq in sequences])
        except Exception as e:
            raise ValueError(f"批量特征提取失败: {str(e)}")
    
    def predict_region_batch(self, data: pd.DataFrame, gb_threshold: float = 0.1) -> pd.DataFrame:
        """Deprecated compatibility wrapper for the public region predictor.

        Uses the central 33 bp of Long_Sequence/399bp through predict_prf;
        feature extraction itself does not own classification models.
        """
        import warnings
        from .. import predict_prf
        warnings.warn('Use PRFPredictor.predict_regions() or predict_prf(data=...) instead',
                      DeprecationWarning, stacklevel=2)
        return predict_prf(data=data, short_threshold=gb_threshold)

    def extract_features(self, sequence: str) -> list:
        """Return the trained feature dimensions after trimming or N-padding the input.

        This shares the preprocessing used by batch feature extraction, including
        U-to-T normalization. Short inputs are right-padded to ``seq_length``.
        Feature extraction errors are raised rather than replaced by zero vectors.
        """
        return self._preprocess_sequence(sequence)
