import numpy as np
from typing import List, Union

class CNNInputProcessor:
    """CNN模型输入数据处理器"""
    
    def __init__(self, max_length: int = 399):
        self.max_length = max_length
        self.base_to_num = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4} 
    
    def trim_sequence(self, seq, target_length):
        """Center-crop to exactly target_length, removing an odd extra base on the right."""
        if len(seq) <= target_length:
            return seq
        start = (len(seq) - target_length) // 2
        return seq[start:start + target_length]
    
    def prepare_sequence(self, sequence: str) -> np.ndarray:
        """Encode an exactly sized input; U is treated as T and unknown bases as N."""
        sequence = str(sequence).upper().replace('U', 'T')
        sequence = self.trim_sequence(sequence, self.max_length)
        encoded = [self.base_to_num.get(base, 4) for base in sequence]
        encoded.extend([4] * (self.max_length - len(encoded)))
        return np.array(encoded).reshape(1, self.max_length, 1)
