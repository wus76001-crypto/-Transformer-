# 小型 Encoder-Decoder Transformer（训练与生成）

本项目实现了一个**可训练的 Encoder-Decoder Transformer**，用于**下一个 token 预测**与简单文本生成，
并内置了**消融实验**、训练指标记录与可视化。

> 代码入口：`train.py`（训练）与 `test.py`（推理）；模型结构在 `model.py`。

---

## 环境要求

- Python ≥ 3.9
- 建议使用 **CUDA GPU** 或 **Apple Silicon (MPS)** 以加速训练
- 安装 PyTorch（根据你的 CUDA 版本选择合适的轮子）：<https://pytorch.org/get-started/locally/>

```bash
# 先安装与你机器匹配的 PyTorch（示例：CPU 或 CUDA 版本）
# 请到 PyTorch 官网选择命令；若用 CPU，可简单：
pip install --index-url https://download.pytorch.org/whl/cpu torch
# 然后安装其余依赖
pip install -r requirements.txt
```

## 数据准备

- `train.py` 期望 **一个本地纯文本文件**（UTF-8），如 `data.txt`。
- 程序会：读取整份文本 → 使用 HuggingFace `AutoTokenizer` 编码 → 按 `--valid_ratio` 切分为训练/验证 →
  再按 `--block_size` 切成定长块进行语言建模训练（移位预测）。

> 中文推荐分词器：`bert-base-chinese`；英文可用 `gpt2`。

## 快速开始

1）安装依赖：
```bash
pip install -r requirements.txt
```

2）准备数据文件：
```bash
# 举例：将你的语料放到 data.txt
echo "Hello world." > data.txt
```

3）一键脚本（默认输出到 `results/` 并进行推理演示）：
```bash
bash scripts/run.sh -d data.txt -t gpt2 -e 3 -b 16 -B 256 -s results
```

- 训练完成后，脚本会把 `results/baseline/best.pt` 复制为 `results/best.pt`，
  以满足 `test.py` 的默认加载路径。
- `test.py` 默认的提示词是 Shakespeare 片段，可自行打开文件修改 `prompt` 文本。

## 直接使用命令行训练（不走脚本）

```bash
python train.py       --data_path data.txt       --tokenizer_name gpt2 \ 
  --block_size 256       --n_embd 128 --n_heads 4 --n_layer 3 --dropout 0.1       --batch_size 32 --epochs 3       --lr 3e-5 --weight_decay 0.01       --warmup_steps 200       --eval_interval 200       --save_dir results       --amp
```

- 训练过程会在 `results/baseline/` 下保存：
  - `best.pt`：最佳验证 PPL 的权重（包含 `model_args` 与使用的 `tokenizer` 名称）
  - `metrics.json`：训练/验证指标
  - `metrics_plot_baseline.png`：损失、PPL、学习率、梯度范数曲线

## 推理与生成

- 训练好后：
```bash
# 确保存在 results/best.pt（脚本已自动处理；若手动训练，请复制一份）
cp results/baseline/best.pt results/best.pt  # 如路径不同请自行调整
python test.py
```
- `test.py` 会：
  1. 从 `results/best.pt` 复现模型结构与分词器
  2. 使用 `generate()` （温度/top-p、重复惩罚、n-gram 去重等）生成文本
  3. 在终端打印输入与输出

> 如需自定义提示词，请编辑 `test.py` 中的 `prompt` 字符串。

## 消融实验

- 已内置的标签：`baseline`、`no_dropout`、`smaller_model`、`fewer_layers`、`fewer_heads`、
  `no_warmup`、`higher_lr`、`lower_lr`。
- 运行单个实验：
```bash
python train.py --data_path data.txt --tokenizer_name gpt2 --ablation no_dropout
```
- 一次性跑一组：
```bash
python train.py --data_path data.txt --tokenizer_name gpt2 --run_all_ablations
# 或者指定列表
python train.py --data_path data.txt --tokenizer_name gpt2 --ablation_list baseline smaller_model fewer_layers
```
- 结果与图表将分别保存到 `results/<实验名>/` 与 `results/ablation_comparison.png`。

## 目录结构（关键项）

```text
model.py        # 模型结构与模块
train.py        # 训练/评估/消融/可视化
test.py         # 推理与生成（从 best.pt 加载）
scripts/run.sh  # 一键训练+推理脚本
requirements.txt
README.md
results/    # 训练输出目录（首次运行后生成）
data.txt        # 你的训练语料（示例）
```

## 常见问题

- **Q: CUDA 版本不匹配 / 不能导入 `torch`？**  
  A: 请到 PyTorch 官网按你的系统/驱动选择正确的安装命令；本项目不强制固定版本。

- **Q: `bert-base-chinese` 报缺少 `sentencepiece`？**  
  A: `transformers` 会自动拉取依赖；若需要可手动安装：`pip install sentencepiece`。

- **Q: `test.py` 加载不到权重？**  
  A: 确保存在 `results/best.pt`。如果你把最佳模型保存在其他目录，可复制或软链接过来。


