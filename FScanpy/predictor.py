import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from .features.sequence import SequenceFeatureExtractor
from .features.cnn_input import CNNInputProcessor
from .utils import extract_window_sequences
import matplotlib.pyplot as plt
import joblib


class PRFPredictor:

    def __init__(self, model_dir=None):
        """
        初始化PRF预测器
        
        Args:
            model_dir: 模型目录路径（可选）
        """
        if model_dir is None:
            model_dir = Path(__file__).resolve().parent / 'pretrained'
        
        try:
            # 设备
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 加载模型 - 使用新的命名约定
            self.short_model = self._load_pickle(os.path.join(model_dir, 'short.pkl'))  # HistGB模型

            # 优先使用 PyTorch 权重 long.pth；若不存在则回退到 long.pkl（兼容旧版本）
            long_pth = os.path.join(model_dir, 'long.pth')
            long_pkl = os.path.join(model_dir, 'long.pkl')
            if os.path.exists(long_pth):
                self.long_model = self._load_long_torch(long_pth)
            else:
                self.long_model = self._load_pickle(long_pkl)
            
            # 初始化特征提取器和CNN处理器，使用与训练时相同的序列长度
            self.short_seq_length = 33   # HistGB使用的序列长度
            self.long_seq_length = 399   # BiLSTM-CNN使用的序列长度
            
            # 初始化特征提取器和CNN输入处理器
            self.feature_extractor = SequenceFeatureExtractor(seq_length=self.short_seq_length)
            self.cnn_processor = CNNInputProcessor(max_length=self.long_seq_length)
            
            # 检测模型类型以优化预测性能
            self._detect_model_types()
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"无法找到模型文件: {str(e)}。请确保 'short.pkl' 存在，且优先使用 'long.pth'（若无则需要 'long.pkl'） 位于 {model_dir}")
        except Exception as e:
            raise Exception(f"加载模型出错: {str(e)}")
    
    def _load_pickle(self, path):
        """安全加载pickle文件"""
        try:
            return joblib.load(path)
        except Exception as e:
            raise FileNotFoundError(f"无法加载模型文件 {path}: {str(e)}")

    # ===== PyTorch Long 模型实现（与 internal_train_pytorch.py 对齐） =====
    class _BiLSTM_CNN_Model(nn.Module):
        def __init__(self, input_dim=1, embedding_dim=64, lstm_units=64, cnn_filters=64,
                     kernel_sizes=[3, 5, 7], dropout_rate=0.5, sequence_length=399):
            super(PRFPredictor._BiLSTM_CNN_Model, self).__init__()
            self.sequence_length = sequence_length
            self.conv_layers = nn.ModuleList()
            for kernel_size in kernel_sizes:
                conv = nn.Sequential(
                    nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters,
                              kernel_size=kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(cnn_filters),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2)
                )
                self.conv_layers.append(conv)
            cnn_output_length = sequence_length // 2
            total_cnn_features = len(kernel_sizes) * cnn_filters * cnn_output_length
            self.lstm1 = nn.LSTM(input_dim, lstm_units, batch_first=True, bidirectional=True)
            self.bn_lstm1 = nn.BatchNorm1d(sequence_length)
            self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units // 2, batch_first=True, bidirectional=True)
            self.bn_lstm2 = nn.BatchNorm1d(lstm_units)
            total_features = lstm_units + total_cnn_features
            self.fc1 = nn.Linear(total_features, 256)
            self.bn_fc1 = nn.BatchNorm1d(256)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(256, 1)
        def forward(self, x):
            batch_size = x.size(0)
            x_cnn = x.permute(0, 2, 1)
            cnn_outputs = []
            for conv in self.conv_layers:
                conv_out = conv(x_cnn)
                conv_out = conv_out.view(batch_size, -1)
                cnn_outputs.append(conv_out)
            cnn_merged = torch.cat(cnn_outputs, dim=1)
            lstm_out, _ = self.lstm1(x)
            lstm_out = self.bn_lstm1(lstm_out)
            lstm_out, _ = self.lstm2(lstm_out)
            lstm_out = lstm_out[:, -1, :]
            lstm_out = self.bn_lstm2(lstm_out)
            merged = torch.cat([lstm_out, cnn_merged], dim=1)
            out = self.fc1(merged)
            out = self.bn_fc1(out)
            out = torch.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            out = torch.sigmoid(out)
            return out.squeeze()

    def _load_long_torch(self, checkpoint_path):
        """加载 PyTorch long 模型权重"""
        model = PRFPredictor._BiLSTM_CNN_Model(
            input_dim=1,
            embedding_dim=64,
            lstm_units=64,
            cnn_filters=64,
            kernel_sizes=[3, 5, 7],
            dropout_rate=0.5,
            sequence_length=399
        ).to(self.device)
        # 兼容在不同设备保存/加载
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        # 兼容 weights_only=True/False 的保存
        if isinstance(state, dict) and all(k.startswith('module.') for k in state.keys()):
            # 去掉分布式前缀
            state = {k.replace('module.', '', 1): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
        return model
    
    def _detect_model_types(self):
        """检测模型类型以优化预测性能"""
        self.short_is_sklearn = hasattr(self.short_model, 'predict_proba')
        self.long_is_sklearn = hasattr(self.long_model, 'predict_proba')
        try:
            import torch as _t
            self.long_is_torch = isinstance(self.long_model, nn.Module)
        except Exception:
            self.long_is_torch = False
    
    def _predict_model(self, model, features, is_sklearn, seq_length):
        """统一的模型预测方法"""
        try:
            if is_sklearn:
                # sklearn模型使用特征向量
                if isinstance(features, np.ndarray) and features.ndim > 1:
                    features = features.flatten()
                features_2d = np.array([features])
                pred = model.predict_proba(features_2d)
                return pred[0][1]
            else:
                # 深度学习模型（Keras 旧分支）
                # 保留向后兼容，但 long 模型若为 torch 将不走此分支
                if seq_length == self.long_seq_length:
                    model_input = self.cnn_processor.prepare_sequence(features)
                else:
                    base_to_num = {'A': 1, 'T': 2, 'G': 3, 'C': 4, 'N': 0}
                    seq_numeric = [base_to_num.get(base, 0) for base in features.upper()]
                    model_input = np.array(seq_numeric).reshape(1, len(seq_numeric), 1)
                try:
                    pred = model.predict(model_input, verbose=0)
                except TypeError:
                    pred = model.predict(model_input)
                if isinstance(pred, list):
                    pred = pred[0]
                if hasattr(pred, 'shape') and len(pred.shape) > 1 and pred.shape[1] > 1:
                    return pred[0][1]
                else:
                    return pred[0][0] if hasattr(pred[0], '__getitem__') else pred[0]
                    
        except Exception as e:
            raise Exception(f"模型预测失败: {str(e)}")

    def predict_single_position(self, fs_period, full_seq, short_threshold=0.1, ensemble_weight=0.4):
        '''
        预测单个位置的PRF状态
        
        Args:
            fs_period: 33bp序列 (short模型使用)
            full_seq: 完整序列 (long模型使用)
            short_threshold: short模型的概率阈值 (默认为0.1)
            ensemble_weight: short模型在集成中的权重 (默认为0.4，long权重为0.6)
        Returns:
            dict: 包含预测概率的字典
        '''
        try:
            # 验证权重参数
            if not (0.0 <= ensemble_weight <= 1.0):
                raise ValueError("ensemble_weight 必须在 0.0 到 1.0 之间")
            
            long_weight = 1.0 - ensemble_weight
            
            # 处理序列长度
            if len(fs_period) > self.short_seq_length:
                fs_period = self.feature_extractor.trim_sequence(fs_period, self.short_seq_length)
                
            # Short模型预测 (HistGB)
            try:
                if self.short_is_sklearn:
                    short_features = self.feature_extractor.extract_features(fs_period)
                    short_prob = self._predict_model(self.short_model, short_features, True, self.short_seq_length)
                else:
                    short_prob = self._predict_model(self.short_model, fs_period, False, self.short_seq_length)
            except Exception as e:
                print(f"Short模型预测时出错: {str(e)}")
                short_prob = 0.0
            
            # 如果short概率低于阈值，则跳过long模型
            if short_prob < short_threshold:
                return {
                    'Short_Probability': short_prob,
                    'Long_Probability': 0.0,
                    'Ensemble_Probability': 0.0,
                    'Ensemble_Weights': f'Short:{ensemble_weight:.1f}, Long:{long_weight:.1f}'
                }

            # Long模型预测 (BiLSTM-CNN)
            try:
                if getattr(self, 'long_is_torch', False):
                    long_prob = self._predict_long_torch(full_seq)
                elif self.long_is_sklearn:
                    long_features = self.feature_extractor.extract_features(full_seq)
                    long_prob = self._predict_model(self.long_model, long_features, True, self.long_seq_length)
                else:
                    long_prob = self._predict_model(self.long_model, full_seq, False, self.long_seq_length)
            except Exception as e:
                print(f"Long模型预测时出错: {str(e)}")
                long_prob = 0.0

            # 计算集成概率
            try:
                ensemble_prob = ensemble_weight * short_prob + long_weight * long_prob
            except Exception as e:
                print(f"计算集成概率时出错: {str(e)}")
                ensemble_prob = (short_prob + long_prob) / 2
            
            return {
                'Short_Probability': short_prob,
                'Long_Probability': long_prob,
                'Ensemble_Probability': ensemble_prob,
                'Ensemble_Weights': f'Short:{ensemble_weight:.1f}, Long:{long_weight:.1f}'
            }
            
        except Exception as e:
            raise Exception(f"预测过程出错: {str(e)}")

    # ===== Torch long 预测路径 =====
    @staticmethod
    def _process_sequence(seq):
        seq = str(seq).upper()
        return ''.join('N' if base not in 'ATCG' else base for base in seq)

    @staticmethod
    def _encode_sequence(seq, max_length=399):
        vocab_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
        encoded = [vocab_map.get(base, 4) for base in seq]
        if len(encoded) > max_length:
            encoded = encoded[:max_length]
        else:
            encoded += [4] * (max_length - len(encoded))
        return np.array(encoded, dtype=np.float32).reshape(1, max_length, 1)

    def _predict_long_torch(self, full_seq):
        processed = PRFPredictor._process_sequence(full_seq)
        encoded = PRFPredictor._encode_sequence(processed, max_length=self.long_seq_length)
        x = torch.from_numpy(encoded).to(self.device)
        with torch.no_grad():
            prob = self.long_model(x).detach().cpu().numpy().reshape(-1)[0]
        # 保证在 [0,1]
        prob = float(np.clip(prob, 0.0, 1.0))
        return prob
    
    def predict_sequence(self, sequence, window_size=3, short_threshold=0.1, ensemble_weight=0.4):
        """
        预测完整序列中的PRF位点（滑动窗口方法）
        
        Args:
            sequence: 输入DNA序列
            window_size: 滑动窗口大小 (默认为3)
            short_threshold: short模型概率阈值 (默认为0.1)
            ensemble_weight: short模型在集成中的权重 (默认为0.4)
            
        Returns:
            pd.DataFrame: 包含预测结果的DataFrame
        """
        if window_size < 1:
            raise ValueError("窗口大小必须大于等于1")
        if short_threshold < 0:
            raise ValueError("short模型阈值必须大于等于0")
        if not (0.0 <= ensemble_weight <= 1.0):
            raise ValueError("ensemble_weight 必须在 0.0 到 1.0 之间")
        
        results = []
        long_weight = 1.0 - ensemble_weight
        
        try:
            # 确保序列为字符串并转换为大写
            sequence = str(sequence).upper()
            
            # 滑动窗口预测
            for pos in range(0, len(sequence) - 2, window_size):
                # 提取窗口序列
                fs_period, full_seq = extract_window_sequences(sequence, pos)
                
                if fs_period is None or full_seq is None:
                    continue
                
                # 预测并记录结果
                pred = self.predict_single_position(fs_period, full_seq, short_threshold, ensemble_weight)
                pred.update({
                    'Position': pos,
                    'Codon': sequence[pos:pos+3],
                    'Short_Sequence': fs_period,  # 更清晰的命名
                    'Long_Sequence': full_seq     # 更清晰的命名
                })
                results.append(pred)
            
            # 创建结果DataFrame
            results_df = pd.DataFrame(results)
            
            return results_df
            
        except Exception as e:
            raise Exception(f"序列预测过程出错: {str(e)}")
    
    def plot_sequence_prediction(self, sequence, window_size=3, short_threshold=0.65, 
                                long_threshold=0.8, ensemble_weight=0.4, title=None, save_path=None, 
                                figsize=(12, 8), dpi=300):
        """
        Plot sequence PRF prediction results
        
        Args:
            sequence: Input DNA sequence
            window_size: Sliding window size (default: 3)
            short_threshold: Short model (HistGB) filtering threshold (default: 0.65)
            long_threshold: Long model (BiLSTM-CNN) filtering threshold (default: 0.8)
            ensemble_weight: Weight of short model in ensemble (default: 0.4)
            title: Plot title (optional)
            save_path: Save path (optional, saves plot if provided)
            figsize: Figure size (default: (12, 8))
            dpi: Figure resolution (default: 300)
            
        Returns:
            tuple: (pd.DataFrame, matplotlib.figure.Figure) prediction results and figure object
        """
        try:
            # Validate weight parameter
            if not (0.0 <= ensemble_weight <= 1.0):
                raise ValueError("ensemble_weight must be between 0.0 and 1.0")
            
            long_weight = 1.0 - ensemble_weight
            
            # Get prediction results
            results_df = self.predict_sequence(sequence, window_size=window_size, 
                                             short_threshold=0.1, ensemble_weight=ensemble_weight)
            
            if results_df.empty:
                raise ValueError("Prediction results are empty, please check input sequence")
            
            # Get sequence length
            seq_length = len(sequence)
            
            # Calculate display width
            desired_visual_width = max(3, seq_length // 100)  # FS site width ~1% of sequence length
            prob_width = max(1, desired_visual_width // 3)    # Prediction probability width is 1/3 of FS site width
            
            # Create figure with three subplots, set height ratios
            fig = plt.figure(figsize=figsize)
            
            # Set title
            if title:
                fig.suptitle(title, y=0.95, fontsize=10)
            else:
                fig.suptitle(f'PRF Prediction Results (Weights {ensemble_weight:.1f}:{long_weight:.1f})', y=0.95, fontsize=10)
            
            # Adjust subplot ratios, make top two heatmaps smaller
            gs = fig.add_gridspec(3, 1, height_ratios=[0.1, 0.1, 1], hspace=0.2)
            
            # FS site heatmap - using fixed width, no blur effect
            ax0 = fig.add_subplot(gs[0])
            fs_data = np.zeros((1, seq_length))
            # Note: No actual FS site information in sliding window prediction, so keep empty or show predicted sites
            # Show high-confidence predictions as potential FS sites
            for _, row in results_df.iterrows():
                pos = int(row['Position'])
                if (row['Short_Probability'] >= short_threshold and 
                    row['Long_Probability'] >= long_threshold and
                    row['Ensemble_Probability'] >= 0.8):  # High confidence threshold
                    half_width = desired_visual_width // 2
                    start_pos = max(0, pos - half_width)
                    end_pos = min(seq_length, pos + half_width + 1)
                    fs_data[0, start_pos:end_pos] = 1  # Use fixed value, no gradient
                
            ax0.imshow(fs_data, cmap='Reds', aspect='auto', interpolation='nearest')
            ax0.set_xticks([])
            ax0.set_yticks([])
            ax0.set_title('FS site', pad=2, fontsize=8)
            
            # Prediction probability heatmap - using fixed width to display probabilities
            ax1 = fig.add_subplot(gs[1])
            prob_data = np.zeros((1, seq_length))
            
            # Apply dual threshold filtering
            for _, row in results_df.iterrows():
                pos = int(row['Position'])
                if (row['Short_Probability'] >= short_threshold and 
                    row['Long_Probability'] >= long_threshold):
                    # Set fixed width for each probability value
                    start = max(0, pos - prob_width//2)
                    end = min(seq_length, pos + prob_width//2 + 1)
                    prob_data[0, start:end] = row['Ensemble_Probability']
            
            im = ax1.imshow(prob_data, cmap='Reds', aspect='auto', vmin=0, vmax=1, interpolation='nearest')
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title('Prediction', pad=2, fontsize=8)
            
            # Main plot (bar chart)
            ax2 = fig.add_subplot(gs[2])
            
            # Apply filtering thresholds
            filtered_probs = results_df['Ensemble_Probability'].copy()
            mask = ((results_df['Short_Probability'] < short_threshold) | 
                   (results_df['Long_Probability'] < long_threshold))
            filtered_probs[mask] = 0
            
            # Draw bar chart - use black color and alpha=0.6 to match prediction_sample style
            ax2.bar(results_df['Position'], filtered_probs, 
                    alpha=0.6, color='black', width=1.0)
            
            # Set x-axis ticks
            step = max(seq_length // 10, 50)
            ax2.set_xticks(np.arange(0, seq_length, step))
            ax2.tick_params(axis='x', rotation=45)
            
            # Set labels
            ax2.set_xlabel('Position')
            ax2.set_ylabel('Probability')
            
            # Set y-axis range
            ax2.set_ylim(0, 1)
            
            # Add grid
            ax2.grid(True, alpha=0.3)
            
            # Ensure all subplots have consistent x-axis range
            for ax in [ax0, ax1, ax2]:
                ax.set_xlim(-1, seq_length)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot if save path is provided
            if save_path:
                save_path = os.fspath(save_path)
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                # Also save PDF version
                if save_path.endswith('.png'):
                    pdf_path = save_path.replace('.png', '.pdf')
                    plt.savefig(pdf_path, bbox_inches='tight')
                print(f"Plot saved to: {save_path}")
            
            return results_df, fig
            
        except Exception as e:
            raise Exception(f"Error plotting sequence prediction: {str(e)}")
    
    def predict_regions(self, sequences, short_threshold=0.1, ensemble_weight=0.4):
        '''
        Predict region sequences (batch prediction of known 399bp sequences)
        
        Args:
            sequences: 399bp sequences or DataFrame/Series/list containing 399bp sequences
            short_threshold: Short model probability threshold (default: 0.1)
            ensemble_weight: Weight of short model in ensemble (default: 0.4)
            
        Returns:
            DataFrame: DataFrame containing prediction probabilities for all sequences
        ''' 
        try:
            # Validate weight parameter
            if not (0.0 <= ensemble_weight <= 1.0):
                raise ValueError("ensemble_weight must be between 0.0 and 1.0")
            
            # Unify input format
            if isinstance(sequences, pd.DataFrame):
                if 'Long_Sequence' in sequences.columns:
                    sequences = sequences['Long_Sequence']
                elif '399bp' in sequences.columns:
                    sequences = sequences['399bp']
                else:
                    raise ValueError("DataFrame must contain 'Long_Sequence' or '399bp' column")
            if isinstance(sequences, pd.Series):
                sequences = sequences.tolist()
            elif isinstance(sequences, str):
                sequences = [sequences]
            
            results = []
            for i, seq399 in enumerate(sequences):
                try:
                    # Extract central 33bp from 399bp sequence (for short model use)
                    seq33 = self._extract_center_sequence(seq399, target_length=self.short_seq_length)
                    
                    # Use unified prediction method
                    pred_result = self.predict_single_position(seq33, seq399, short_threshold, ensemble_weight)
                    pred_result.update({
                        'Short_Sequence': seq33,
                        'Long_Sequence': seq399
                    })
                    
                    results.append(pred_result)
                    
                except Exception as e:
                    print(f"Error processing sequence {i+1}: {str(e)}")
                    long_weight = 1.0 - ensemble_weight
                    results.append({
                        'Short_Probability': 0.0,
                        'Long_Probability': 0.0,
                        'Ensemble_Probability': 0.0,
                        'Ensemble_Weights': f'Short:{ensemble_weight:.1f}, Long:{long_weight:.1f}',
                        'Short_Sequence': self._extract_center_sequence(seq399, target_length=self.short_seq_length) if len(seq399) >= self.short_seq_length else seq399,
                        'Long_Sequence': seq399
                    })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            raise Exception(f"Error in region prediction process: {str(e)}")

    def _extract_center_sequence(self, sequence, target_length=33):
        """Extract subsequence of specified length from center position of sequence"""
        # Ensure sequence is string
        sequence = str(sequence).upper()
        
        # If sequence length is less than target length, return original sequence
        if len(sequence) <= target_length:
            return sequence
        
        # Calculate center position
        center = len(sequence) // 2
        half_target = target_length // 2
        
        # Extract center sequence
        start = center - half_target
        end = start + target_length
        
        # Boundary check
        if start < 0:
            start = 0
            end = target_length
        elif end > len(sequence):
            end = len(sequence)
            start = end - target_length
        
        return sequence[start:end]

    # 兼容性方法（向后兼容，但标记为废弃）
    def predict_full(self, sequence, window_size=3, short_threshold=0.1, short_weight=0.4, plot=False):
        """
        ⚠️ 已废弃：请使用 predict_sequence() 方法
        
        向后兼容的方法，内部调用新的 predict_sequence()
        """
        import warnings
        warnings.warn("predict_full() 已废弃，请使用 predict_sequence() 方法", DeprecationWarning, stacklevel=2)
        
        # 调用新方法并添加兼容性字段
        results_df = self.predict_sequence(sequence, window_size, short_threshold, short_weight)
        
        # 添加兼容性字段
        if 'Ensemble_Probability' in results_df.columns:
            results_df['Voting_Probability'] = results_df['Ensemble_Probability']
            results_df['Weighted_Probability'] = results_df['Ensemble_Probability']
        if 'Ensemble_Weights' in results_df.columns:
            results_df['Weight_Info'] = results_df['Ensemble_Weights']
        if 'Short_Sequence' in results_df.columns:
            results_df['33bp'] = results_df['Short_Sequence']
        if 'Long_Sequence' in results_df.columns:
            results_df['399bp'] = results_df['Long_Sequence']
        
        if plot:
            # 如果需要绘图，调用绘图方法
            _, fig = self.plot_sequence_prediction(sequence, window_size, 0.65, 0.8, short_weight)
            return results_df, fig
        
        return results_df
    
    def predict_region(self, seq, short_threshold=0.1, short_weight=0.4):
        """
        ⚠️ 已废弃：请使用 predict_regions() 方法
        
        向后兼容的方法，内部调用新的 predict_regions()
        """
        import warnings
        warnings.warn("predict_region() 已废弃，请使用 predict_regions() 方法", DeprecationWarning, stacklevel=2)
        
        # 调用新方法并添加兼容性字段
        results_df = self.predict_regions(seq, short_threshold, short_weight)
        
        # 添加兼容性字段
        if 'Ensemble_Probability' in results_df.columns:
            results_df['Voting_Probability'] = results_df['Ensemble_Probability']
            results_df['Weighted_Probability'] = results_df['Ensemble_Probability']
        if 'Ensemble_Weights' in results_df.columns:
            results_df['Weight_Info'] = results_df['Ensemble_Weights']
        if 'Short_Sequence' in results_df.columns:
            results_df['33bp'] = results_df['Short_Sequence']
        if 'Long_Sequence' in results_df.columns:
            results_df['399bp'] = results_df['Long_Sequence']
        
        return results_df
