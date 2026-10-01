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
from ._validation import positive_integer, probability
from .plotting import plot_prediction_results
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
        """Predict one position; propagate model errors instead of inventing zero scores."""
        short_threshold = probability(short_threshold, 'short_threshold')
        ensemble_weight = probability(ensemble_weight, 'ensemble_weight')
        long_weight = 1.0 - ensemble_weight
        if len(fs_period) > self.short_seq_length:
            fs_period = self.feature_extractor.trim_sequence(fs_period, self.short_seq_length)
        try:
            if self.short_is_sklearn:
                features = self.feature_extractor.extract_features(fs_period)
                short_prob = self._predict_model(self.short_model, features, True, self.short_seq_length)
            else:
                short_prob = self._predict_model(self.short_model, fs_period, False, self.short_seq_length)
            short_prob = probability(short_prob, 'Short model probability')
        except Exception as exc:
            raise RuntimeError(f'Short model prediction failed: {exc}') from exc
        weights = f'Short:{ensemble_weight:.1f}, Long:{long_weight:.1f}'
        if short_prob < short_threshold:
            return {'Short_Probability': short_prob, 'Long_Probability': 0.0,
                    'Ensemble_Probability': 0.0, 'Ensemble_Weights': weights}
        try:
            if getattr(self, 'long_is_torch', False):
                long_prob = self._predict_long_torch(full_seq)
            elif self.long_is_sklearn:
                features = self.feature_extractor.extract_features(full_seq)
                long_prob = self._predict_model(self.long_model, features, True, self.long_seq_length)
            else:
                long_prob = self._predict_model(self.long_model, full_seq, False, self.long_seq_length)
            long_prob = probability(long_prob, 'Long model probability')
        except Exception as exc:
            raise RuntimeError(f'Long model prediction failed: {exc}') from exc
        return {'Short_Probability': short_prob, 'Long_Probability': long_prob,
                'Ensemble_Probability': ensemble_weight * short_prob + long_weight * long_prob,
                'Ensemble_Weights': weights}

    # ===== Torch long 预测路径 =====
    @staticmethod
    def _process_sequence(seq):
        seq = str(seq).upper().replace('U', 'T')
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
        """Scan every ``window_size`` nucleotides, retaining codon-aligned model inputs.

        Model input lengths stay at 33 and 399 bp. A per-position scan preserves
        every requested output row but reuses inference for repeated codon windows.
        """
        window_size = positive_integer(window_size, 'window_size')
        short_threshold = probability(short_threshold, 'short_threshold')
        ensemble_weight = probability(ensemble_weight, 'ensemble_weight')
        if sequence is None:
            raise ValueError('sequence must be a nucleotide sequence')
        sequence = str(sequence).upper()
        results = []
        last_frame, last_prediction = None, None
        for pos in range(0, len(sequence) - 2, window_size):
            frame = pos - pos % 3
            if frame != last_frame:
                fs_period, full_seq = extract_window_sequences(sequence, pos)
                try:
                    last_prediction = self.predict_single_position(fs_period, full_seq, short_threshold, ensemble_weight)
                except Exception as exc:
                    raise RuntimeError(f'Prediction failed at Position {pos}: {exc}') from exc
                last_frame = frame
            pred = dict(last_prediction)
            pred.update({'Position': pos, 'Codon': sequence[pos:pos + 3],
                         'Short_Sequence': fs_period, 'Long_Sequence': full_seq})
            results.append(pred)
        columns = ['Short_Probability', 'Long_Probability', 'Ensemble_Probability', 'Ensemble_Weights',
                   'Position', 'Codon', 'Short_Sequence', 'Long_Sequence']
        return pd.DataFrame(results, columns=columns)
    
    def plot_sequence_prediction(self, sequence, window_size=3, short_threshold=0.65,
                                 long_threshold=0.8, ensemble_weight=0.4, title=None, save_path=None,
                                 figsize=(12, 8), dpi=300, *, reference_positions=None,
                                 heatmap_ratios=(0.1, 0.1, 1), candidate_threshold=0.8):
        """Predict and plot with the original two-heatmap/bar layout.

        Existing positional arguments and return values are unchanged. Display
        thresholds filter the plot; the inference gate is lowered when necessary
        so a short display threshold below 0.1 can expose the long-model score.
        Optional reference_positions uses independently supplied 0-based coordinates.
        Set heatmap_ratios=(0.35,0.35,2.8) for thick tutorial-style heatmaps.
        """
        short_threshold = probability(short_threshold, 'short_threshold')
        long_threshold = probability(long_threshold, 'long_threshold')
        ensemble_weight = probability(ensemble_weight, 'ensemble_weight')
        results = self.predict_sequence(sequence, window_size=window_size,
                                        short_threshold=min(0.1, short_threshold),
                                        ensemble_weight=ensemble_weight)
        return plot_prediction_results(
            results, sequence_length=len(str(sequence)), short_threshold=short_threshold,
            long_threshold=long_threshold, title=title or f'PRF Prediction Results (Weights {ensemble_weight:.1f}:{1-ensemble_weight:.1f})',
            save_path=save_path, figsize=figsize, dpi=dpi, reference_positions=reference_positions,
            heatmap_ratios=heatmap_ratios, candidate_threshold=candidate_threshold)
    
    def predict_regions(self, sequences, short_threshold=0.1, ensemble_weight=0.4):
        """Predict region sequences; invalid rows and model failures raise with their index."""
        short_threshold = probability(short_threshold, 'short_threshold')
        ensemble_weight = probability(ensemble_weight, 'ensemble_weight')
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
        for index, region in enumerate(sequences):
            if not isinstance(region, str) or not region:
                raise ValueError(f'Region sequence {index + 1} must be a nonempty nucleotide string')
            short = self._extract_center_sequence(region, target_length=self.short_seq_length)
            try:
                result = self.predict_single_position(short, region, short_threshold, ensemble_weight)
            except Exception as exc:
                raise RuntimeError(f'Prediction failed for region {index + 1}: {exc}') from exc
            result.update({'Short_Sequence': short, 'Long_Sequence': region})
            results.append(result)
        columns = ['Short_Probability', 'Long_Probability', 'Ensemble_Probability', 'Ensemble_Weights',
                   'Short_Sequence', 'Long_Sequence']
        return pd.DataFrame(results, columns=columns)

    def extract_features(self, sequences):
        """Return a 2D short-model feature array for a string or iterable of strings."""
        if isinstance(sequences, str):
            sequences = [sequences]
        features = self.feature_extractor.extract_features_batch(sequences)
        return features.reshape(-1, len(self.feature_extractor.feature_names))

    def get_model_info(self):
        """Describe the loaded model types and their effective input lengths."""
        backend = 'pytorch' if self.long_is_torch else ('sklearn' if self.long_is_sklearn else 'keras')
        return {'short_model': type(self.short_model).__name__,
                'long_model': type(self.long_model).__name__, 'backend': backend,
                'short_input_bp': self.short_seq_length,
                'long_input_bp': self.short_seq_length if self.long_is_sklearn else self.long_seq_length}

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
