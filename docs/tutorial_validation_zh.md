# 新版 notebook 与安装包验证

新版中英文预测教程已改用包内的 `PRFPredictor.plot_sequence_prediction()` 与 `plot_prediction_regions()`，通过 `get_test_data_path("predict_sample_examples.csv")` 读取随包数据。原有 Markdown 描述逐单元格核对，保持不变；旧版参考 notebook 与作者版本逐字节一致。

测试使用构建后安装的 FScanpy 1.0.1 wheel。每个 notebook 在新内核中运行，工作目录位于仓库之外，没有 `data` 或 `tutorial` 目录。验证代码同时检查实际导入位置与数据文件位置，确认均来自安装包；发布 notebook 已删除这些诊断单元格。

| Notebook | 成功执行的代码单元格 | 已保存图像 | 结果 |
|---|---:|---:|---|
| `FScanpy_Demo.ipynb` | 9 | 1 | 通过 |
| `tutorial/predict_sample.ipynb` | 8 | 10 | 通过 |
| `tutorial/predict_sample_zh.ipynb` | 8 | 10 | 通过 |
| `examples/reusable_plotting.ipynb` | 4 | 3 | 通过 |

源码与安装包均通过 55 项测试。测试涵盖示例数据的字段、序列长度、参考坐标、峰值位置、局部高分数量，以及每隔 1/3/6 nt 扫描的一致性。整段图保留上方两条粗红色热图、下方黑色柱状图；局部比较使用统一横轴范围与概率刻度。

五条教学序列随 wheel 一起分发。表中的坐标均为 0-based 核苷酸坐标，峰值是在短/长模型显示阈值均为 0.2、短模型权重 0.6 时得到的结果。

| Sequence_ID | 序列长度（nt） | 参考起点 | 全序列最高峰位置 | 最高显示分数 |
|---:|---:|---:|---:|---:|
| 0 | 780 | 297 | 300 | 0.963053 |
| 1 | 705 | 309 | 645 | 0.976088 |
| 2 | 423 | 249 | 249 | 0.999480 |
| 3 | 759 | 543 | 24 | 0.997771 |
| 4 | 1089 | 252 | 258 | 0.969608 |

数据仅包含序列标识、名称、功能描述、长度、参考坐标和完整序列，未加入 PRF ID、split 或训练/测试来源字段。原 `full_seq.xlsx` 等既有数据仍保留，新教程与可复用绘图示例统一使用新的 CSV。

在 notebook 所用的 Python 内核中安装本次候选包并重启内核后，可将新版 notebook 单独放在任意目录运行。1.0.1 目前仍是未发布候选版，需要安装本 PR 的源码或相应 wheel；旧的已发布包不包含本次新增绘图函数和示例 CSV。

复验入口为 `tests/run_notebooks.py SOURCE OUTPUT`，其中 `OUTPUT` 应为仓库外的新目录；测试需要安装 `nbclient`、`nbformat` 和 `ipykernel`，无需复制教程数据。
