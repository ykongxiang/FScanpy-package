import numpy as np
from typing import List, Union

class CNNInputProcessor:
    """CNN模型输入数据处理器"""
    
    def __init__(self, max_length: int = 399):
        self.max_length = max_length
        self.base_to_num = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4} 
    
    def trim_sequence(self, seq, target_length):
        """
        从序列两端等量截取，使其达到目标长度，保持中心位置不变

        参数:
        seq: 原始序列
        target_length: 目标长度

        返回:
        截取后的序列
        """
        if len(seq) <= target_length:
            return seq

        # 计算需要从每端截取的长度
        excess = len(seq) - target_length
        trim_each_side = excess // 2

        # 从两端等量截取，保持中心位置不变
        return seq[trim_each_side:len(seq)-trim_each_side]
    
    def prepare_sequence(self, sequence: str) -> np.ndarray:
        """
        处理单个序列
        
        Args:
            sequence: DNA序列
                
        Returns:
            np.ndarray: 处理后的序列数组
        """
        try:
            # 序列验证和预处理
            if not isinstance(sequence, str):
                sequence = str(sequence)
            
            sequence = sequence.upper().replace('U', 'T')
            
            # 如果序列长度不等于目标长度，进行截取
            if len(sequence) > self.max_length:
                sequence = self.trim_sequence(sequence, self.max_length)
                
            # 使用与训练时相同的编码方式
            self.base_to_num = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}  # 与SemiBilstmCnn.py中保持一致
            
            # 序列转换为数字
            seq_numeric = []
            for base in sequence:
                seq_numeric.append(self.base_to_num.get(base, 4))  # 未知碱基用4表示
            
            # 填充序列
            if len(seq_numeric) < self.max_length:
                seq_numeric.extend([4] * (self.max_length - len(seq_numeric)))
            
            # 重塑数据为三维数组 (samples, timesteps, features)
            result = np.array(seq_numeric).reshape(1, self.max_length, 1)
            
            # 检查结果维度
            if result.ndim != 3:
                print(f"警告: CNN输入维度异常 - {result.ndim}，应为3")
                # 强制修正为正确的维度
                result = result.reshape(1, self.max_length, 1)
                
            return result
                
        except Exception as e:
            print(f"CNN序列处理失败: {str(e)}")
            # 出错时返回全零的三维数组
            return np.zeros((1, self.max_length, 1))