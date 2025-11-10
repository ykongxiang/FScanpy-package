import numpy as np
import pandas as pd
from typing import Tuple, Optional
from Bio import SeqIO
from Bio.Seq import Seq

def fscanr(blastx_output: pd.DataFrame, 
           mismatch_cutoff: float = 10,
           evalue_cutoff: float = 1e-5,
           frameDist_cutoff: float = 10) -> pd.DataFrame:
    """
    identify PRF sites from BLASTX output
    
    Args:
        blastx_output: BLASTX output DataFrame
        mismatch_cutoff: mismatch threshold
        evalue_cutoff: E-value threshold 
        frameDist_cutoff: frame distance threshold
        
    Returns:
        pd.DataFrame: DataFrame containing PRF site information
    """
    blastx = blastx_output.copy()
    
    blastx.columns = ["qseqid", "sseqid", "pident", "length", "mismatch", 
                     "gapopen", "qstart", "qend", "sstart", "send", 
                     "evalue", "bitscore", "qframe", "sframe"]
    
    blastx = blastx[
        (blastx['evalue'] <= evalue_cutoff) & 
        (blastx['mismatch'] <= mismatch_cutoff)
    ].dropna()

    freq = blastx['qseqid'].value_counts()
    multi_hits = freq[freq > 1].index
    blastx = blastx[blastx['qseqid'].isin(multi_hits)]
    
    blastx = blastx.sort_values(['qseqid', 'sseqid', 'qstart'])
    
    prf_list = []
    for i in range(1, len(blastx)):
        curr = blastx.iloc[i]
        prev = blastx.iloc[i-1]
        
        if (curr['qseqid'] == prev['qseqid'] and 
            curr['sseqid'] == prev['sseqid'] and
            curr['qframe'] != prev['qframe'] and 
            curr['qframe'] * prev['qframe'] > 0):
            
            if curr['qframe'] > 0 and prev['qframe'] > 0:
                frame_start = prev['qend'] 
                frame_end = curr['qstart']
                pep_start = prev['send']
                pep_end = curr['sstart']
                strand = "+"
            elif curr['qframe'] < 0 and prev['qframe'] < 0:
                frame_start = prev['qstart']
                frame_end = curr['qend'] 
                pep_start = curr['send']
                pep_end = prev['sstart']
                strand = "-"
            else:
                continue
                
            q_dist = frame_end - frame_start - 1
            s_dist = pep_end - pep_start
            fs_type = q_dist + (1 - s_dist) * 3
            
            if (abs(q_dist) <= frameDist_cutoff and 
                abs(s_dist) <= frameDist_cutoff // 3 and
                -3 < fs_type < 3):
                
                prf_list.append({
                    'DNA_seqid': curr['qseqid'],
                    'FS_start': frame_start,
                    'FS_end': frame_end,
                    'Pep_seqid': curr['sseqid'],
                    'Pep_FS_start': prev['send'] + 1,
                    'Pep_FS_end': curr['sstart'],
                    'FS_type': fs_type,
                    'Strand': strand
                })
    
    if not prf_list:
        print("No PRF events detected!")
        return pd.DataFrame()
        
    prf = pd.DataFrame(prf_list)
    
    for col in ['DNA_seqid', 'Pep_seqid']:
        for pos in ['FS_start', 'FS_end']:
            loci = prf[col] + '_' + prf[pos].astype(str)
            prf = prf[~loci.duplicated()]
            
    return prf

def extract_prf_regions(mrna_file: str, prf_data: pd.DataFrame) -> pd.DataFrame:
    """
    从mRNA序列中提取PRF位点周围的序列
    
    Args:
        mrna_file: mRNA序列文件路径 (FASTA格式)
        prf_data: FScanR输出的PRF位点数据
        
    Returns:
        pd.DataFrame: 包含399bp序列的DataFrame
    """
    mrna_dict = {rec.id: str(rec.seq) 
                 for rec in SeqIO.parse(mrna_file, "fasta")}
    
    results = []
    for _, row in prf_data.iterrows():
        seq_id = row['DNA_seqid']
        if seq_id not in mrna_dict:
            print(f"警告: {seq_id} 未在mRNA文件中找到")
            continue
            
        sequence = mrna_dict[seq_id]
        strand = row['Strand']
        fs_start = int(row['FS_start'])
        
        try:
            if strand == '-':
                sequence = str(Seq(sequence).reverse_complement())
            
            # 只提取399bp序列，33bp由predictor内部截取
            full_seq = extract_window_sequences(sequence, fs_start)[1]
            
            results.append({
                'DNA_seqid': seq_id,
                'FS_start': fs_start,
                'FS_end': int(row['FS_end']),
                'Strand': strand,
                '399bp': full_seq,
                'FS_type': row['FS_type']
            })
            
        except Exception as e:
            print(f"处理 {seq_id} 时出错: {str(e)}")
            continue
            
    return pd.DataFrame(results)

def extract_window_sequences(seq: str, position: int) -> Tuple[Optional[str], Optional[str]]:
    """
    从指定位置提取分析窗口序列
    
    Args:
        seq: 输入DNA序列
        position: 当前分析位置 (FS_start)
    
    Returns:
        Tuple[str, str]: (33bp序列, 399bp序列) - 已调整为与训练模型匹配的长度
    """
    # 确保位置在密码子边界上（整数倍的3）
    frame_position = position - (position % 3)

    # 计算33bp窗口的起止位置 (GB模型)
    half_size_small = 33 // 2
    start_small = frame_position - half_size_small
    end_small = frame_position + half_size_small + (33 % 2)  # 添加余数以处理奇数长度
    
    # 计算399bp窗口的起止位置 (CNN模型)
    half_size_large = 399 // 2
    start_large = frame_position - half_size_large
    end_large = frame_position + half_size_large + (399 % 2)  # 添加余数以处理奇数长度

    # 提取序列并填充
    seq_small = _extract_and_pad(seq, start_small, end_small, 33)
    seq_large = _extract_and_pad(seq, start_large, end_large, 399)
    
    return seq_small, seq_large

def _extract_and_pad(seq: str, start: int, end: int, target_length: int) -> str:
    """提取序列并用N填充"""
    if start < 0:
        prefix = 'N' * abs(start)
        extracted = prefix + seq[:end]
    elif end > len(seq):
        suffix = 'N' * (end - len(seq))
        extracted = seq[start:] + suffix
    else:
        extracted = seq[start:end]
    
    # 确保序列长度正确
    if len(extracted) < target_length:
        # 从中心填充
        pad_left = (target_length - len(extracted)) // 2
        pad_right = target_length - len(extracted) - pad_left
        extracted = 'N' * pad_left + extracted + 'N' * pad_right
    elif len(extracted) > target_length:
        # 从序列两端等量截取
        excess = len(extracted) - target_length
        trim_each_side = excess // 2
        extracted = extracted[trim_each_side:len(extracted)-trim_each_side]
    
    return extracted

def prepare_cnn_input(sequence: str) -> np.ndarray:
    """prepare CNN model input"""
    base_to_num = {'A': 1, 'T': 2, 'G': 3, 'C': 4, 'N': 0}
    seq_numeric = [base_to_num.get(base, 0) for base in sequence.upper()]
    return np.array(seq_numeric).reshape(1, len(sequence), 1) 