## Abstract
FScanpy is a Python package designed to predict Programmed Ribosomal Frameshifting (PRF) sites in DNA sequences. It integrates advanced machine learning models, including Gradient Boosting and BiLSTM-CNN, to provide accurate predictions. This tool is essential for understanding gene expression regulation in various organisms, including eukaryotes and viruses, and offers a robust solution for PRF prediction challenges.

## Introduction
![FScanpy structure](/tutorial/image/structure.jpeg)

FScanpy is a Python package dedicated to predicting Programmed Ribosomal Frameshifting (PRF) sites in DNA sequences. It integrates machine learning models (Gradient Boosting and BiLSTM-CNN) along with the FScanR package to furnish precise PRF predictions. Users are capable of employing three types of data as input: the entire cDNA/mRNA sequence that requires prediction, the nucleotide sequence in the vicinity of the suspected frameshift site, and the peptide library blastx results of the species or related species. It anticipates the input sequence to be in the + strand and can be integrated with FScanR to augment the accuracy.

![Machine learning models](/tutorial/image/ML.png)
For the prediction of the entire sequence, FScanpy adopts a sliding window approach to scan the entire sequence and predict the PRF sites. For regional prediction, it is based on the 30-bp and 300-bp sequences in the 0 reading frame around the suspected frameshift site. Initially, the Gradient Boosting model will predict the potential PRF sites within the scanning window. If the predicted probability exceeds the threshold, the BiLSTM-CNN model will predict the PRF sites in the 300bp sequence.Then,VotingClassifier will combine the two models to make the final prediction.

For PRF detection from BLASTX output, FScanpy identifies potential PRF sites from BLASTX alignment results, acquires the two hits of the same query sequence, and then utilizes frameDist_cutoff, mismatch_cutoff, and evalue_cutoff to filter the hits. Finally, it employs [FScanR](https://github.com/seanchen607/FScanR.git) to identify the PRF sites.

### Background
[Ribosomal frameshifting](https://en.wikipedia.org/wiki/Ribosomal_frameshift), also known as translational frameshifting or translational recoding, is a biological phenomenon that occurs during translation that results in the production of multiple, unique proteins from a single mRNA. The process can be programmed by the nucleotide sequence of the mRNA and is sometimes affected by the secondary, 3-dimensional mRNA structure. It has been described mainly in viruses (especially retroviruses), retrotransposons and bacterial insertion elements, and also in some cellular genes.

### Key features of FScanpy include:

- Integration of two predictive models:
  - [Gradient Boosting](https://tensorflow.google.cn/tutorials/estimator/boosted_trees?hl=en): Analyzes local sequence features centered around potential frameshift sites (10 codons).
  - [BiLSTM-CNN](https://paperswithcode.com/method/cnn-bilstm): Analyzes broader sequence features (100 codons).
- Supports PRF prediction across various species.
- Can be combined with [FScanR](https://github.com/seanchen607/FScanR.git) for enhanced accuracy.

## Installation (python>=3.7)

### 1. Use pip
```bash
pip install FScanpy
```

### 2. Clone from [GitHub](https://github.com/ykongxiang/FScanpy.git)
```bash
git clone https://github.com/ykongxiang/FScanpy.git
cd your_project_directory
pip install -e .
```

## Methods and Usage

### 1. Load model and test data
Test data can be found in `FScanpy/data/test_data`,you can use the `list_test_data()` method to list all the test data and the `get_test_data_path()` method to get the path of the test data:
```python
from FScanpy import PRFPredictor
from FScanpy.data import get_test_data_path, list_test_data
predictor = PRFPredictor() # load model
list_test_data() # list all the test data
blastx_file = get_test_data_path('blastx_example.xlsx')
mrna_file = get_test_data_path('mrna_example.fasta')
region_example = get_test_data_path('region_example.xlsx')
```

### 2. Predict PRF Sites in a Full Sequence
Use the `predict_full()` method to scan the entire sequence,you can use the `window_size` parameter to adjust the scanning window size(default is 3) and the `gb_threshold` parameter to adjust the Gradient Boosting model fitting threshold(default is 0.1) for faster or more accurate prediction:
```python
    '''
    Args:
        sequence: mRNA sequence
        window_size: scanning window size (default is 3)
        gb_threshold: Gradient Boosting model threshold (default is 0.1)
    Returns:
        results: DataFrame containing prediction probabilities
    ''' 
results = predictor.predict_full(sequence='ATGCGTACGTATGCGTACGTATGCGTACGT',
                               window_size=3,    # Scanning window size
                               gb_threshold=0.1, # Gradient Boosting model threshold
                               plot=True) # Whether to plot the prediction results
fig.savefig('predict_full.png')
```

### 3. Predict PRF in Specific Regions
Use the `predict_region()` method to predict PRF in known regions of interest:
```python
    '''
    Args:
        seq_30bp: 30bp sequence
            seq_300bp: 300bp sequence
            gb_threshold: GB model probability threshold (default is 0.1)
        Returns:
        dict: dictionary containing prediction probabilities
    '''
region_example = pd.read_excel(get_test_data_path('region_example.xlsx'))
results = predictor.predict_region(seq_30bp=region_example['30bp'],  # Local sequence (30bp)
                                 seq_300bp=region_example['300bp']) # Context sequence (300bp)
```

### 4. Identify PRF Sites from BLASTX Output
BLASTX Output should contain the following columns: `qseqid`, `sseqid`, `pident`, `length`, `mismatch`, `gapopen`, `qstart`, `qend`, `sstart`, `send`, `evalue`, `bitscore`, `qframe`, `sframe`.

FScanR result contains `DNA_seqid`, `FS_start`, `FS_end`, `FS_type`,`Pep_seqid`, `Pep_FS_start`, `Pep_FS_end`, `Strand` columns.
Use the FScanR function to identify potential PRF sites from BLASTX alignment results:
```python
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
from FScanpy.utils import fscanr
blastx_output = pd.read_excel(get_test_data_path('blastx_example.xlsx'))
fscanr_result = fscanr(blastx_output, 
                      mismatch_cutoff=10,    # Allowed mismatches
                      evalue_cutoff=1e-5,    # E-value threshold
                      frameDist_cutoff=10)   # Frame distance threshold
```

### 5. Extract PRF Sites from BLASTX Output or your Sequence Data
Use the `extract_prf_regions()` method to extract PRF site sequences from mRNA sequences,it based on the `FS_start` column of the FScanR output contact with the `DNA_seqid` column of the input mRNA sequence file to extract the 30bp and 300bp sequences around the PRF sites in 0 reading frame:
```python
    """
    extract PRF site sequences from mRNA sequences
    
    Args:
        mrna_file: mRNA sequence file path (FASTA format)
        prf_data: FScanR output PRF site data or your suspected PRF site data which at least contains `DNA_seqid` `FS_start` `strand` columns
        
    Returns:
        pd.DataFrame: DataFrame containing 30bp and 300bp sequences
    """
from FScanpy.utils import extract_prf_regions
prf_regions = extract_prf_regions(mrna_file=get_test_data_path('mrna_example.fasta'),
                                prf_data=fscanr_result)
```


## Citation
If you use FScanpy, please cite our paper: [Paper Link] 