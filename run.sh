#!/usr/bin/env bash
    set -euo pipefail

    # 默认参数
    DATA="data.txt"
    TOKENIZER="gpt2"
    EPOCHS=3
    BATCH=32
    BLOCK=256
    SAVE_DIR="checkpoints"
    EVAL_INT=200
    AMP_FLAG="--amp"   # 如果你想禁用 AMP，可设置为空字符串

    usage() {
      cat <<EOF
用法: $(basename "$0") [选项]

  -d  数据文件路径 (默认: data.txt)
  -t  分词器名称，如 gpt2 / bert-base-chinese (默认: gpt2)
  -e  训练 epoch 数 (默认: 3)
  -b  训练 batch size (默认: 32)
  -B  block_size 序列长度 (默认: 256)
  -s  保存目录 save_dir (默认: checkpoints)
  -i  评估间隔 steps (默认: 200)
  -A  关闭 AMP (混合精度)，加此参数则禁用

示例：
  bash scripts/run.sh -d data.txt -t gpt2 -e 3 -b 16 -B 256 -s checkpoints
EOF
    }

    # 解析参数
    while getopts ":d:t:e:b:B:s:i:A" opt; do
      case $opt in
        d) DATA="$OPTARG" ;;
        t) TOKENIZER="$OPTARG" ;;
        e) EPOCHS="$OPTARG" ;;
        b) BATCH="$OPTARG" ;;
        B) BLOCK="$OPTARG" ;;
        s) SAVE_DIR="$OPTARG" ;;
        i) EVAL_INT="$OPTARG" ;;
        A) AMP_FLAG="" ;;
        \?) echo "未知参数: -$OPTARG" ; usage ; exit 1 ;;
        :)  echo "选项 -$OPTARG 缺少参数" ; usage ; exit 1 ;;
      esac
    done

    echo "==> 开始训练"
    python train.py       --data_path "$DATA"       --tokenizer_name "$TOKENIZER"       --block_size "$BLOCK"       --batch_size "$BATCH"       --epochs "$EPOCHS"       --lr 3e-5       --weight_decay 0.01       --warmup_steps 200       --eval_interval "$EVAL_INT"       --save_dir "$SAVE_DIR"       ${AMP_FLAG}

    echo "==> 规范化 best 权重路径以供 test.py 使用"
    mkdir -p "$SAVE_DIR"
    if [ -f "$SAVE_DIR/baseline/best.pt" ]; then
      cp -f "$SAVE_DIR/baseline/best.pt" "$SAVE_DIR/best.pt"
      echo "已复制 $SAVE_DIR/baseline/best.pt -> $SAVE_DIR/best.pt"
    elif [ -f "$SAVE_DIR/best.pt" ]; then
      echo "检测到 $SAVE_DIR/best.pt"
    else
      echo "未找到 best.pt，请检查训练输出。"
    fi

    # test.py 默认从 checkpoints/best.pt 读取
    if [ "$SAVE_DIR" != "checkpoints" ]; then
      echo "==> 将 $SAVE_DIR/best.pt 同步到 checkpoints/best.pt 以匹配 test.py 默认路径"
      mkdir -p checkpoints
      if [ -f "$SAVE_DIR/best.pt" ]; then
        cp -f "$SAVE_DIR/best.pt" checkpoints/best.pt
      fi
    fi

    echo "==> 运行推理示例"
    python test.py
