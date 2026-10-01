import os
from pathlib import Path
from numbers import Integral
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from .features.sequence import SequenceFeatureExtractor
from .features.cnn_input import CNNInputProcessor
from .utils import extract_window_sequences, positive_integer, probability
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
            start = (len(encoded) - max_length) // 2
            encoded = encoded[start:start + max_length]
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
        """Center-crop with the same odd-length convention as both model inputs."""
        sequence = str(sequence).upper()
        if len(sequence) <= target_length:
            return sequence
        start = (len(sequence) - target_length) // 2
        return sequence[start:start + target_length]

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


PROBABILITY_COLUMNS = ['Short_Probability', 'Long_Probability', 'Ensemble_Probability']


def _table(results):
    if not isinstance(results, pd.DataFrame):
        raise ValueError('results must be a prediction DataFrame for one sequence')
    required = ['Position'] + PROBABILITY_COLUMNS
    missing = [name for name in required if name not in results.columns]
    if missing:
        raise ValueError(f'Missing prediction columns: {missing}')
    if results.empty:
        raise ValueError('Prediction results are empty')
    positions = results.Position.to_numpy(dtype=float)
    scores = results[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(positions).all() or (positions < 0).any() or (positions != np.floor(positions)).any():
        raise ValueError('Position must contain nonnegative integer nucleotide coordinates')
    if results.Position.duplicated().any():
        raise ValueError('Position must be unique; pass results for one sequence at a time')
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError('Prediction probabilities must be finite and between 0 and 1')
    return positions.astype(int)


def _scores(results, short_threshold, long_threshold):
    short_threshold = probability(short_threshold, 'short_threshold')
    long_threshold = probability(long_threshold, 'long_threshold')
    mask = (results.Short_Probability >= short_threshold) & (results.Long_Probability >= long_threshold)
    return results.Ensemble_Probability.where(mask, 0).to_numpy(dtype=float), mask.to_numpy()


def _ratios(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('heatmap_ratios must contain three positive finite numbers')
    return values.tolist()


def _references(values, length=None):
    if values is None:
        return []
    if np.isscalar(values):
        values = [values]
    references = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0 or (length is not None and value >= length):
            raise ValueError('reference_positions must contain 0-based integer coordinates within the sequence')
        references.append(int(value))
    return references


def _save(figure, save_path, dpi):
    if save_path is not None:
        path = Path(save_path)
        figure.savefig(path, dpi=dpi, bbox_inches='tight')
        if path.suffix.lower() == '.png':
            figure.savefig(path.with_suffix('.pdf'), bbox_inches='tight')


def plot_prediction_results(results, sequence_length=None, short_threshold=0.65,
                            long_threshold=0.8, title=None, save_path=None,
                            figsize=(12, 8), dpi=300, *, reference_positions=None,
                            heatmap_ratios=(0.1, 0.1, 1), candidate_threshold=0.8):
    """Plot an existing single-sequence prediction table; return ``(results, figure)``.

    The original two red heatmaps and black bars are retained. Use
    ``heatmap_ratios=(0.35, 0.35, 2.8)`` for the tutorial's thick heatmaps.
    ``reference_positions`` marks independently supplied 0-based positions in
    green; candidate heatmaps remain generated from the prediction scores.
    Overlapping heatmap marks show their maximum score, not the last row's score.
    ``sequence_length`` defaults to the last scanned position plus three; pass
    the actual length to retain the full extent of a sparsely scanned sequence.
    The input table is not modified and no models are loaded.
    """
    positions = _table(results)
    length = positive_integer(sequence_length if sequence_length is not None else int(positions.max()) + 3,
                              'sequence_length')
    if positions.max() >= length:
        raise ValueError('sequence_length must be larger than every scanned Position')
    scores, eligible = _scores(results, short_threshold, long_threshold)
    candidate_threshold = probability(candidate_threshold, 'candidate_threshold')
    ratios = _ratios(heatmap_ratios)
    references = _references(reference_positions, length)

    desired_width = max(3, length // 100)
    probability_width = max(1, desired_width // 3)
    candidates = np.zeros((1, length))
    heatmap_scores = np.zeros((1, length))
    for position, score, visible in zip(positions, scores, eligible):
        if not visible:
            continue
        start, end = max(0, position - probability_width // 2), min(length, position + probability_width // 2 + 1)
        heatmap_scores[0, start:end] = np.maximum(heatmap_scores[0, start:end], score)
        if score >= candidate_threshold:
            start, end = max(0, position - desired_width // 2), min(length, position + desired_width // 2 + 1)
            candidates[0, start:end] = 1

    figure = plt.figure(figsize=figsize, dpi=dpi)
    figure.suptitle(title or 'PRF Prediction Results', fontsize=10)
    grid = figure.add_gridspec(3, 1, height_ratios=ratios)
    axes = [figure.add_subplot(grid[row]) for row in range(3)]
    for axis, data, label in zip(axes[:2], [candidates, heatmap_scores], ['FS site (predicted candidates)', 'Prediction']):
        axis.imshow(data, cmap='Reds', aspect='auto', vmin=0, vmax=1, interpolation='nearest')
        axis.set(xticks=[], yticks=[], title=label)
        axis.title.set_fontsize(10)
    bars = axes[2]
    bars.bar(positions, scores, alpha=0.6, color='black', width=1)
    bars.set(xlabel='Position (0-based nt)', ylabel='Filtered ensemble score', ylim=(0, 1))
    bars.set_xticks(np.arange(0, length, max(length // 10, 50)))
    bars.tick_params(axis='x', rotation=45)
    bars.grid(True, alpha=0.3)
    for axis in axes:
        axis.set_xlim(-1, length)
        for reference in references:
            axis.axvline(reference, color='#009E73', linestyle='--', linewidth=1.2,
                         label=f'Reference {reference}' if axis is bars else None)
    if references:
        bars.legend(loc='upper right', fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    _save(figure, save_path, dpi)
    return results, figure


def plot_prediction_regions(results, centers, radius=15, short_threshold=0.2,
                            long_threshold=0.2, score_threshold=0.7, *,
                            candidate_threshold=0.8, reference_positions=None,
                            heatmap_ratios=(0.35, 0.35, 2.8), figsize=None,
                            dpi=120, save_path=None):
    """Compare equally sized regions of existing predictions, without inference.

    ``centers`` is a list of 0-based positions or ``(label, position)`` pairs,
    e.g. ``[('Reference', 309), ('Competitor', 645)]``. Return ``(summary, figure)``.
    Each column contains two thick heatmaps above bars with the same 0–1 scale.
    ``high_score_positions`` counts scanned rows reaching ``score_threshold``;
    nearby scores can share overlapping input windows and are not independent
    observations. Regions with no scanned rows have NaN ``local_max``.
    """
    positions = _table(results)
    scores, eligible = _scores(results, short_threshold, long_threshold)
    if isinstance(radius, bool) or not isinstance(radius, Integral) or radius < 0:
        raise ValueError('radius must be a nonnegative integer in nucleotides')
    radius = int(radius)
    score_threshold = probability(score_threshold, 'score_threshold')
    candidate_threshold = probability(candidate_threshold, 'candidate_threshold')
    ratios = _ratios(heatmap_ratios)
    references = _references(reference_positions)
    regions = []
    for item in centers:
        if isinstance(item, Integral) and not isinstance(item, bool):
            label, center = 'Region', item
        else:
            label, center = item
        if isinstance(center, bool) or not isinstance(center, Integral) or center < 0:
            raise ValueError('centers must contain nonnegative integer nucleotide coordinates')
        regions.append((str(label), int(center)))
    if not regions:
        raise ValueError('Provide at least one region center')

    figure = plt.figure(figsize=figsize or (max(5, 3.5 * len(regions)), 5.3), dpi=dpi)
    grid = figure.add_gridspec(3, len(regions), height_ratios=ratios)
    summaries = []
    for column, (label, center) in enumerate(regions):
        local = np.abs(positions - center) <= radius
        count = int(np.sum(local & eligible & (scores >= score_threshold)))
        summaries.append(dict(region=label, center=center, high_score_positions=count,
                              scanned_positions=int(local.sum()),
                              local_max=float(scores[local].max()) if local.any() else float('nan')))
        width = 2 * radius + 1
        candidates, heatmap_scores = np.zeros((1, width)), np.zeros((1, width))
        for position, score, visible in zip(positions[local], scores[local], eligible[local]):
            if not visible:
                continue
            index = int(position - center + radius)
            heatmap_scores[0, index] = score
            if score >= candidate_threshold:
                candidates[0, max(0, index - 1):min(width, index + 2)] = 1
        axes = [figure.add_subplot(grid[row, column]) for row in range(3)]
        extent = (-radius - 0.5, radius + 0.5, 0, 1)
        for axis, data in zip(axes[:2], [candidates, heatmap_scores]):
            axis.imshow(data, cmap='Reds', aspect='auto', vmin=0, vmax=1,
                        interpolation='nearest', extent=extent)
            axis.set(xticks=[], yticks=[])
        axes[0].set_title(f'{label}: {center}\nFS site (predicted candidates)', fontsize=9)
        axes[1].set_title('Prediction', fontsize=9)
        bars = axes[2]
        bars.bar(positions[local] - center, scores[local], color='black', alpha=0.6, width=1)
        bars.axhline(score_threshold, color='gray', linestyle=':', linewidth=1)
        bars.set(xlim=extent[:2], ylim=(0, 1.03), xlabel='Offset from center (nt)',
                 title=f'Score >= {score_threshold:g}: {count} positions')
        bars.set_xticks([-radius, 0, radius] if radius else [0])
        bars.grid(axis='y', alpha=0.2)
        if column == 0:
            bars.set_ylabel('Filtered ensemble score')
        for axis in axes:
            for reference in references:
                if abs(reference - center) <= radius:
                    axis.axvline(reference - center, color='#009E73', linestyle='--', linewidth=1.2)
    figure.tight_layout()
    _save(figure, save_path, dpi)
    return pd.DataFrame(summaries), figure
