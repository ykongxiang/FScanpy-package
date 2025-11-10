from .predictor import PRFPredictor
from . import data
import pandas as pd
import numpy as np
from typing import Union, List, Dict

__version__ = '0.3.0'
__author__ = ''
__email__ = ''

__all__ = ['PRFPredictor', 'predict_prf', 'plot_prf_prediction', 'data', '__version__', '__author__', '__email__']

def predict_prf(
    sequence: Union[str, List[str], None] = None,
    data: Union[pd.DataFrame, None] = None,
    window_size: int = 3,
    short_threshold: float = 0.1,
    ensemble_weight: float = 0.4,
    model_dir: str = None
) -> pd.DataFrame:
    """
    PRF site prediction function
    
    Args:
        sequence: Single or multiple DNA sequences for sliding window prediction
        data: DataFrame data, must contain 'Long_Sequence' or '399bp' column for region prediction
        window_size: Sliding window size (default: 3)
        short_threshold: Short model (HistGB) probability threshold (default: 0.1)
        ensemble_weight: Weight of short model in ensemble (default: 0.4, long weight: 0.6)
        model_dir: Model directory path (optional)
    
    Returns:
        pandas.DataFrame: Prediction results containing the following main fields:
            - Short_Probability: Short model prediction probability
            - Long_Probability: Long model prediction probability  
            - Ensemble_Probability: Ensemble prediction probability (main result)
            - Ensemble_Weights: Weight configuration information
        
    Examples:
        # 1. Single sequence sliding window prediction
        >>> from FScanpy import predict_prf
        >>> sequence = "ATGCGTACGT..."
        >>> results = predict_prf(sequence=sequence)
        
        # 2. Multiple sequences sliding window prediction
        >>> sequences = ["ATGCGTACGT...", "GCTATAGCAT..."]
        >>> results = predict_prf(sequence=sequences)
        
        # 3. Custom ensemble weight ratio
        >>> results = predict_prf(sequence=sequence, ensemble_weight=0.3)  # 3:7 ratio
        
        # 4. DataFrame region prediction
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     'Long_Sequence': ['ATGCGT...', 'GCTATAG...']  # or use '399bp'
        ... })
        >>> results = predict_prf(data=data)
    """
    predictor = PRFPredictor(model_dir=model_dir)
    
    # Validate input parameters
    if sequence is None and data is None:
        raise ValueError("Must provide either sequence or data parameter")
    if sequence is not None and data is not None:
        raise ValueError("Cannot provide both sequence and data parameters")
    if not (0.0 <= ensemble_weight <= 1.0):
        raise ValueError("ensemble_weight must be between 0.0 and 1.0")
    
    # Sliding window prediction mode
    if sequence is not None:
        if isinstance(sequence, str):
            # Single sequence prediction
            return predictor.predict_sequence(
                sequence, window_size, short_threshold, ensemble_weight)
        elif isinstance(sequence, (list, tuple)):
            # Multiple sequences prediction
            results = []
            for i, seq in enumerate(sequence, 1):
                try:
                    result = predictor.predict_sequence(
                        seq, window_size, short_threshold, ensemble_weight)
                    result['Sequence_ID'] = f'seq_{i}'
                    results.append(result)
                except Exception as e:
                    print(f"Warning: Sequence {i} prediction failed - {str(e)}")
            return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    # Region prediction mode
    else:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data parameter must be pandas DataFrame type")
        
        # Check column names (support both new and old naming conventions)
        seq_column = None
        if 'Long_Sequence' in data.columns:
            seq_column = 'Long_Sequence'
        elif '399bp' in data.columns:
            seq_column = '399bp'
        else:
            raise ValueError("DataFrame must contain 'Long_Sequence' or '399bp' column")
        
        # Call region prediction function
        try:
            results = predictor.predict_regions(
                data[seq_column], short_threshold, ensemble_weight)
                
            # Add other columns from original data
            for col in data.columns:
                if col not in ['Long_Sequence', '399bp', 'Short_Sequence', '33bp']:
                    results[col] = data[col].values
                    
            return results
            
        except Exception as e:
            print(f"Warning: Region prediction failed - {str(e)}")
            # Create empty results
            long_weight = 1.0 - ensemble_weight
            results = pd.DataFrame({
                'Short_Probability': [0.0] * len(data),
                'Long_Probability': [0.0] * len(data),
                'Ensemble_Probability': [0.0] * len(data),
                'Ensemble_Weights': [f'Short:{ensemble_weight:.1f}, Long:{long_weight:.1f}'] * len(data)
            })
            
            # Add original data columns
            for col in data.columns:
                results[col] = data[col].values
                
            return results

def plot_prf_prediction(
    sequence: str,
    window_size: int = 3,
    short_threshold: float = 0.65,
    long_threshold: float = 0.8,
    ensemble_weight: float = 0.4,
    title: str = None,
    save_path: str = None,
    figsize: tuple = (12, 8),
    dpi: int = 300,
    model_dir: str = None
) -> tuple:
    """
    Plot PRF prediction results for sequence frameshifting probability
    
    Args:
        sequence: Input DNA sequence
        window_size: Sliding window size (default: 3)
        short_threshold: Short model (HistGB) filtering threshold (default: 0.65)
        long_threshold: Long model (BiLSTM-CNN) filtering threshold (default: 0.8)
        ensemble_weight: Weight of short model in ensemble (default: 0.4, long weight: 0.6)
        title: Plot title (optional)
        save_path: Save path (optional, saves plot if provided)
        figsize: Figure size (default: (12, 8))
        dpi: Figure resolution (default: 300)
        model_dir: Model directory path (optional)
    
    Returns:
        tuple: (pd.DataFrame, matplotlib.figure.Figure) prediction results and figure object
        
    Examples:
        # 1. Simple plotting
        >>> from FScanpy import plot_prf_prediction
        >>> sequence = "ATGCGTACGT..."
        >>> results, fig = plot_prf_prediction(sequence)
        >>> plt.show()
        
        # 2. Custom thresholds and ensemble weights
        >>> results, fig = plot_prf_prediction(
        ...     sequence, 
        ...     short_threshold=0.7, 
        ...     long_threshold=0.85,
        ...     ensemble_weight=0.3,  # 3:7 weight ratio
        ...     title="Custom Weight Prediction Results",
        ...     save_path="prediction_result.png"
        ... )
        
        # 3. Equal weight combination
        >>> results, fig = plot_prf_prediction(
        ...     sequence,
        ...     ensemble_weight=0.5  # 5:5 equal weights
        ... )
        
        # 4. Long model dominated
        >>> results, fig = plot_prf_prediction(
        ...     sequence,
        ...     ensemble_weight=0.2  # 2:8 weights, long model dominated
        ... )
    """
    if not (0.0 <= ensemble_weight <= 1.0):
        raise ValueError("ensemble_weight must be between 0.0 and 1.0")
    
    predictor = PRFPredictor(model_dir=model_dir)
    
    return predictor.plot_sequence_prediction(
        sequence=sequence,
        window_size=window_size,
        short_threshold=short_threshold,
        long_threshold=long_threshold,
        ensemble_weight=ensemble_weight,
        title=title,
        save_path=save_path,
        figsize=figsize,
        dpi=dpi
    )