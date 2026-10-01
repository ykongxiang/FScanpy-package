# 复用预测绘图功能

原有 `plot_prf_prediction()` 和 `PRFPredictor.plot_sequence_prediction()` 调用继续可用，已有位置参数和 `(results, figure)` 返回结构保持不变。1.0.1 增加可选的关键字参数：

```python
from FScanpy import plot_prf_prediction

results, figure = plot_prf_prediction(
    sequence,
    window_size=3,
    short_threshold=0.2,
    long_threshold=0.2,
    ensemble_weight=0.6,
    reference_positions=[309],
    heatmap_ratios=(0.35, 0.35, 2.8),
    dpi=120,
)
```

`reference_positions` 接收独立提供的 0-based 核苷酸坐标，以绿色虚线标记。上方候选热图仍由模型分数生成，不以参考注释替代预测。原有默认面板比例 `(0.1, 0.1, 1)` 保留；指定 `(0.35, 0.35, 2.8)` 即可得到预测教程中的两个粗热图。`candidate_threshold` 控制候选条带的集成分数阈值，默认 0.8。

直接复用已有预测表，避免重新加载模型和重新预测：

```python
from FScanpy import plot_prediction_results, plot_prediction_regions

_, figure = plot_prediction_results(
    results, sequence_length=len(sequence),
    short_threshold=0.2, long_threshold=0.2,
    reference_positions=[309],
    heatmap_ratios=(0.35, 0.35, 2.8), dpi=120,
)

summary, figure = plot_prediction_regions(
    results,
    centers=[("Reference", 309), ("Competitor", 645), ("Competitor", 252)],
    radius=15, short_threshold=0.2, long_threshold=0.2,
    score_threshold=0.7, reference_positions=[309],
)
```

两种绘图函数均保留原预测表。局部图使用等宽核苷酸范围和相同概率尺度。统计表包含区域名称 `region`、中心 `center`、达到阈值的扫描位置数 `high_score_positions`、扫描位置总数 `scanned_positions` 及局部最大显示分数 `local_max`。计数指输出行数；重复窗口或相邻重叠窗口不代表独立观测。无扫描位置的区域返回 `local_max=NaN`。

调低显示阈值只会重新过滤已有结果，无法恢复之前被短模型门控跳过的长模型分数。如需这些分数，应调低预测时的 `short_threshold` 后重新计算。原有“预测并绘图”接口在短模型显示阈值低于 0.1 时，会相应降低推理门控。

`window_size` 表示每隔多少个核苷酸扫描一次，模型输入长度仍为 33/399 bp。增大间隔可减少长序列扫描的计算量。现有密码子窗口对齐规则保留；逐位扫描保留重复输出行，并复用对应窗口的预测结果。

完整可运行示例见[已执行的绘图 notebook](../examples/reusable_plotting.ipynb)，使用包内附带序列，无需额外数据路径。

README 中介绍的 `predictor.extract_features(sequences)` 和 `predictor.get_model_info()` 已补齐。前者返回二维短模型特征数组，后者报告模型类型、后端及实际输入长度。遗留的 `SequenceFeatureExtractor.predict_region_batch()` 会发出弃用提示并委托给公开区域接口；推荐直接使用 `PRFPredictor.predict_regions()`。区域输入以 `Long_Sequence` / `399bp` 的中心 33 bp 作为短模型输入。

FScanR 产生的比对坐标沿用 BLASTX 的 1-based 约定。`extract_prf_regions(fasta, sites)` 会先将这些坐标转换到编码链方向的 0-based 位置，再提取窗口；如果输入表已使用 0-based 坐标，请明确传入 `coordinate_base=0`。返回表中的 `FS_start` 和 `FS_end` 保留输入值。参见 [FScanR 原始实现](https://github.com/seanchen607/FScanR/blob/master/R/FScanR.R)和 [NCBI BLASTX 说明](https://blast.ncbi.nlm.nih.gov/Blast.cgi?LINK_LOC=blasthome&PAGE_TYPE=BlastSearch&PROGRAM=blastx)。
