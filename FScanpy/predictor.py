import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from .features.sequence import SequenceFeatureExtractor
from .features.cnn_input import CNNInputProcessor
from .utils import extract_window_sequences
import matplotlib.pyplot as plt


class PRFPredictor:

    def __init__(self, model_dir=None):

        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), 'pretrained')
        
        try:
            self.gb_model = self._load_pickle(os.path.join(model_dir, 'GradientBoosting_all.pkl'))
            self.cnn_model = load_model(os.path.join(model_dir, 'BiLSTM-CNN_all.keras'))
            self.voting_model = self._load_pickle(os.path.join(model_dir, 'Voting_all.pkl'))

            self.feature_extractor = SequenceFeatureExtractor()
            self.cnn_processor = CNNInputProcessor()
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"can't find model file: {str(e)}")
        except Exception as e:
            raise Exception(f"load model error: {str(e)}")
    
    def _load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def predict_single_position(self, fs_period, full_seq, gb_threshold=0.1):
        '''
        Args:
            fs_period: 30bp sequence
            full_seq: 300bp sequence
            gb_threshold: GB model probability threshold (default is 0.1)
        Returns:
            dict: dictionary containing prediction probabilities
        '''
        try:

            gb_features = self.feature_extractor.extract_features(fs_period)
            gb_prob = self.gb_model.predict_proba([gb_features])[0][1]
            
            if gb_prob < gb_threshold:
                return {
                    'GB_Probability': gb_prob,
                    'CNN_Probability': 0.0,
                    'Voting_Probability': 0.0
                }

            cnn_input = self.cnn_processor.prepare_sequence(full_seq)
            cnn_prob = self.cnn_model.predict(cnn_input, verbose=0)[0][0]

            voting_input = np.array([[gb_prob, cnn_prob]])
            voting_prob = self.voting_model.predict_proba(voting_input)[0][1]
            
            return {
                'GB_Probability': gb_prob,
                'CNN_Probability': cnn_prob,
                'Voting_Probability': voting_prob
            }
            
        except Exception as e:
            raise Exception(f"predict process error: {str(e)}")
    
    def predict_full(self, sequence, window_size=3, gb_threshold=0.1, plot=False):
        """
        预测完整序列中的PRF位点
        
        Args:
            sequence: input DNA sequence
            window_size: sliding window size (default is 3)
            gb_threshold: GB model probability threshold (default is 0.1)
            plot: whether to plot the prediction results (default is False)
            
        Returns:
            if plot=False:
                pd.DataFrame: DataFrame containing prediction results
            if plot=True:
                tuple: (pd.DataFrame, matplotlib.figure.Figure)
        """
        if window_size < 1:
            raise ValueError("window size must be greater than or equal to 1")
        if gb_threshold < 0:
            raise ValueError("GB threshold must be greater than or equal to 0")
        
        results = []
        
        try:
            for pos in range(0, len(sequence) - 2, window_size):
                fs_period, full_seq = extract_window_sequences(sequence, pos)
                
                if fs_period is None or full_seq is None:
                    continue
                
                pred = self.predict_single_position(fs_period, full_seq, gb_threshold)
                pred.update({
                    'Position': pos,
                    'Codon': sequence[pos:pos+3],
                    '30bp': fs_period,
                    '300bp': full_seq
                })
                results.append(pred)
            
            results_df = pd.DataFrame(results)
            
            if plot:
                # 创建图形
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
                
                # 绘制折线图
                ax1.plot(results_df['Position'], results_df['GB_Probability'], 
                        label='GB Model', alpha=0.7, linewidth=1.5)
                ax1.plot(results_df['Position'], results_df['CNN_Probability'], 
                        label='CNN Model', alpha=0.7, linewidth=1.5)
                ax1.plot(results_df['Position'], results_df['Voting_Probability'], 
                        label='Voting Model', linewidth=2, color='red')
                
                ax1.set_xlabel('Sequence Position')
                ax1.set_ylabel('Shift Probability')
                ax1.set_title('Shift Prediction Probability')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 准备热图数据
                positions = results_df['Position'].values
                probabilities = results_df['Voting_Probability'].values
                
                # 创建热图矩阵
                heatmap_matrix = np.zeros((1, len(positions)))
                heatmap_matrix[0, :] = probabilities
                
                # 绘制热图
                im = ax2.imshow(heatmap_matrix, aspect='auto', cmap='YlOrRd',
                              extent=[min(positions), max(positions), 0, 1])
                
                # 添加颜色条
                cbar = plt.colorbar(im, ax=ax2)
                cbar.set_label('Shift Probability')
                
                # 设置热图轴标签
                ax2.set_xlabel('Sequence Position')
                ax2.set_title('Shift Probability Heatmap')
                ax2.set_yticks([])
                
                # 调整布局
                plt.tight_layout()
                
                return results_df, fig
            
            return results_df
            
        except Exception as e:
            raise Exception(f"sequence prediction process error: {str(e)}")
    
    def predict_region(self, seq_30bp, seq_300bp, gb_threshold=0.1):
        '''
        predict region sequence
        
        Args:
            seq_30bp: 30bp sequence or DataFrame with 30bp sequences
            seq_300bp: 300bp sequence or DataFrame with 300bp sequences
            gb_threshold: GB model probability threshold (default is 0.1)
            
        Returns:
            DataFrame: DataFrame containing prediction probabilities for all sequences
        ''' 
        try:
            # 如果输入是DataFrame或Series，转换为列表
            if isinstance(seq_30bp, (pd.DataFrame, pd.Series)):
                seq_30bp = seq_30bp.tolist()
            if isinstance(seq_300bp, (pd.DataFrame, pd.Series)):
                seq_300bp = seq_300bp.tolist()
            
            # 如果输入是单个字符串，转换为列表
            if isinstance(seq_30bp, str):
                seq_30bp = [seq_30bp]
            if isinstance(seq_300bp, str):
                seq_300bp = [seq_300bp]
            
            # 确保两个序列列表长度相同
            if len(seq_30bp) != len(seq_300bp):
                raise ValueError("30bp和300bp序列数量不匹配")
            
            results = []
            for i, (seq30, seq300) in enumerate(zip(seq_30bp, seq_300bp)):
                try:
                    # GB模型预测
                    gb_features = self.feature_extractor.extract_features(seq30)
                    gb_prob = self.gb_model.predict_proba([gb_features])[0][1]
                    
                    # 如果GB概率低于阈值，添加低概率结果
                    if gb_prob < gb_threshold:
                        results.append({
                            'GB_Probability': gb_prob,
                            'CNN_Probability': 0.0,
                            'Voting_Probability': 0.0
                        })
                        continue
                    
                    # CNN模型预测
                    cnn_input = self.cnn_processor.prepare_sequence(seq300)
                    cnn_prob = self.cnn_model.predict(cnn_input, verbose=0)[0][0]
                    
                    # 投票模型预测
                    voting_input = np.array([[gb_prob, cnn_prob]])
                    voting_prob = self.voting_model.predict_proba(voting_input)[0][1]
                    
                    results.append({
                        'GB_Probability': gb_prob,
                        'CNN_Probability': cnn_prob,
                        'Voting_Probability': voting_prob
                    })
                    
                except Exception as e:
                    print(f"处理第 {i+1} 个序列时出错: {str(e)}")
                    results.append({
                        'GB_Probability': 0.0,
                        'CNN_Probability': 0.0,
                        'Voting_Probability': 0.0
                    })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            raise Exception(f"区域预测过程出错: {str(e)}")