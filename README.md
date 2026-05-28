# Two-Stream Attention Heatmap Classifier (Late Fusion)

Binary classification task:
- `0 = Distracted`
- `1 = Attentive`

The model consumes two aligned inputs per sample:
1. `frame_XXXXXX.jpg` (RGB lecture frame context)
2. `heatmap_XXXXXX.jpg` (grayscale synthetic gaze distribution)

## Project Files

```text
src/attention_pipeline/
  data_generation.py
  datasets.py
  model.py
  train.py
  evaluate_xai.py
  inference.py
```

## Expected Dataset Layout

```text
data/two_stream_attention/
  train/
    0/
      frame_000001.jpg
      heatmap_000001.jpg
      ...
    1/
      frame_000001.jpg
      heatmap_000001.jpg
      ...
  val/
    0/ ...
    1/ ...
  test/
    0/ ...
    1/ ...
```

## 1) Generate Data From Real Lecture Video

```bash
python -m src.attention_pipeline.data_generation ^
  --video_path lecture.mp4 ^
  --output_dir data/two_stream_attention ^
  --extract_fps 1.0 ^
  --image_size 224 224 ^
  --copies_per_frame 1 ^
  --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
```

## 2) Train Two-Stream CNN

```bash
python -m src.attention_pipeline.train ^
  --data_dir data/two_stream_attention ^
  --epochs 30 ^
  --batch_size 16 ^
  --lr 1e-4 ^
  --output_dir artifacts/two_stream
```

## 3) Evaluate + Heatmap-Branch Grad-CAM

```bash
python -m src.attention_pipeline.evaluate_xai ^
  --data_dir data/two_stream_attention ^
  --weights_path artifacts/two_stream/best_model.pt ^
  --output_dir artifacts/two_stream_eval
```

## 4) Inference Utility

`inference.py` provides:
- `load_trained_model(...)`
- `predict_pair_pil(frame_pil, heatmap_pil, ...)`
- `predict_pair_paths(frame_path, heatmap_path, ...)`

