# FScanpy
## A Machine Learning-Based Framework for Programmed Ribosomal Frameshifting Prediction

FScanpy is a comprehensive Python package designed for the prediction of [Programmed Ribosomal Frameshifting (PRF)](https://en.wikipedia.org/wiki/Ribosomal_frameshift) sites in nucleotide sequences. By integrating advanced machine learning approaches (Gradient Boosting and BiLSTM-CNN) with the established [FScanR](https://github.com/seanchen607/FScanR.git) framework, FScanpy provides robust and accurate PRF site predictions. The package requires input sequences to be in the positive (5' to 3') orientation.

![FScanpy Architecture](/tutorial/image/structure.jpeg)

For detailed documentation and usage examples, please refer to our [tutorial](tutorial/tutorial.md).

## Installation Requirements
- Python ≥ 3.7
- Dependencies are automatically handled during installation

### Option 1: Install via pip
```bash
pip install FScanpy
```

### Option 2: Install from source
```bash
git clone https://github.com/seanchen607/FScanpy-package.git
cd FScanpy-package
pip install -e .
```

## Authors
**Xiao Chen, Professor.**
- Professor, Shandong University, China
- [Personal Profile](https://faculty.sdu.edu.cn/xc/en)

**Yuhao Yang**
- Undergraduate, Shandong University, China

## Citation
If you utilize FScanpy in your research, please cite our work:

```bibtex
[Citation details will be added upon publication]
```
