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
        """
        从序列两端等量截取，使其达到目标长度
        
        Args:
            seq: 原始序列
            target_length: 目标长度
            
        Returns:
            截取后的序列
        """
        if len(seq) <= target_length:
            return seq
            
        # 计算需要从每端截取的长度
        excess = len(seq) - target_length
        trim_each_side = excess // 2
        
        # 从两端等量截取，保持中心位置不变
        return seq[trim_each_side:len(seq)-trim_each_side]
    
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
        """
        批量预测区域序列
        
        Args:
            data: DataFrame包含'33bp'和'399bp'列
            gb_threshold: GB模型概率阈值（默认为0.1）
            
        Returns:
            DataFrame: 包含预测结果的DataFrame
        """
        results = []
        for idx, row in data.iterrows():
            try:
                # 确保序列是字符串
                seq_33bp = str(row['33bp'])
                seq = str(row['399bp'])
                
                # 确保序列长度正确
                seq_33bp = self._preprocess_sequence(seq_33bp)
                seq = self._preprocess_sequence(seq)
                
                # 预测
                result = self.predict_region(seq_33bp, seq, gb_threshold)
                
                # 添加原始数据的其他列
                for col in data.columns:
                    if col not in ['33bp', '399bp']:
                        result[col] = row[col]
                        
                results.append(result)
                
            except Exception as e:
                print(f"处理索引 {idx} 的序列时出错: {str(e)}")
                continue
                
        return pd.DataFrame(results)

    def extract_features(self, sequence: str) -> list:
        """
        提取序列特征
        
        Args:
            sequence: DNA序列
            
        Returns:
            list: 特征向量
        """
        try:
            # 确保输入是字符串
            if not isinstance(sequence, str):
                sequence = str(sequence)
            
            # 大写并替换U为T
            sequence = sequence.upper().replace('U', 'T')
            
            # 如果序列长度不等于目标长度，进行截取
            if len(sequence) != self.seq_length:
                sequence = self.trim_sequence(sequence, self.seq_length)
            
            # 初始化特征列表
            features = []
            
            try:
                # 基础特征 (碱基频率)
                for base in ['A', 'T', 'G', 'C', 'N']:
                    features.append(sequence.count(base) / len(sequence))
                
                # 3-mer特征
                for kmer in [''.join(p) for p in itertools.product(['A', 'T', 'G', 'C', 'N'], repeat=3)]:
                    count = 0
                    for i in range(len(sequence) - 2):
                        if sequence[i:i+3] == kmer:
                            count += 1
                    features.append(count / max(1, len(sequence) - 2))
                
                # 密码子特征
                codons = [''.join(p) for p in itertools.product(['A', 'T', 'G', 'C'], repeat=3)]
                n_codons = len(sequence) // 3
                for i in range(n_codons):
                    pos_start = i * 3
                    current_codon = sequence[pos_start:pos_start+3]
                    for codon in codons:
                        features.append(1 if current_codon == codon and 'N' not in current_codon else 0)
                
                # GC含量
                valid_bases = [b for b in sequence if b != 'N']
                gc_content = (valid_bases.count('G') + valid_bases.count('C')) / len(valid_bases) if valid_bases else 0
                features.append(gc_content)
                
                # 序列复杂度
                from collections import Counter
                valid_counts = Counter(valid_bases)
                total_valid = sum(valid_counts.values())
                entropy = 0
                if total_valid > 0:  # 避免除零错误
                    for cnt in valid_counts.values():
                        if cnt > 0:  # 避免log(0)
                            p = cnt / total_valid
                            entropy += -p * np.log2(p)
                    entropy /= np.log2(4) if len(valid_counts) > 0 else 1  # 归一化到0-1，避免除零
                features.append(entropy)
                
                # 确保返回的是一维列表或数组
                if isinstance(features, np.ndarray) and features.ndim > 1:
                    features = features.flatten()
                
                return features
            
            except Exception as e:
                print(f"特征计算过程出错: {str(e)}")
                # 如果计算过程出错，返回正确长度的全零特征向量
                expected_length = 5 + 125 + (len(sequence) // 3) * 64 + 2  # 根据特征提取逻辑计算特征向量长度
                return [0.0] * expected_length
                
        except Exception as e:
            print(f"特征提取失败: {str(e)}")
            # 返回一个空列表，调用方需处理这种情况
            return []