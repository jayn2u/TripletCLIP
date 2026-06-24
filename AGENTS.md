# TripletCLIP Agent Notes

Use `uv run python` to execute Python code.

## Project role

This repository is the official [TripletCLIP](https://tripletclip.github.io/) release (NeurIPS 2024): improving CLIP compositional reasoning with synthetic vision-language negatives. It is **not** a person ReID project. There is no CUHK-PEDES, ICFG-PEDES, or RSTPReid training pipeline here.

Do not confuse this tree with `lab_clip/train_tripletclip_*`. That code adapts TripletCLIP-style losses to pedestrian ReID; this repo trains on web-scale TripletData in WebDataset tar shards.

## Released code layout

| Path | Role |
|------|------|
| `src/openclip/` | OpenCLIP-based training and evaluation (the supported entry point) |
| `src/openclip/src/main.py` | Train or evaluate |
| `src/openclip/src/data.py` | `TripletClipData` — expects `shard-{0..750}.tar` with image, caption, neg_image, neg_caption |
| `src/openclip/requirements.txt` | Dependencies (conda/venv; no root `pyproject.toml`) |

Original paper training scripts outside `src/openclip` are still listed as TODO in `README.md`.

## Default model and data

- Model: **ViT-B/32** (`--model_name ViT-B-32`)
- Training data: TripletData WebDataset tars under `--data_dir` (CC3M/CC12M scale; not lab ReID JSON)
- Optional validation: MSCOCO via `--val_data_dir`
- Checkpoints saved by training: `{log_dir}/checkpoints/epoch-*.ckpt` with `model_state_dict`

## Pretrained weights (Hugging Face)

Official pretrained checkpoints live on the [TripletCLIP Hugging Face organization](https://huggingface.co/TripletCLIP). They use split `vision-encoder/` and `text-encoder/` safetensors plus a root `utils.py` loader. This format differs from `epoch-*.ckpt` produced by `src/openclip`.

| Model | Hugging Face repo |
|-------|-------------------|
| TripletCLIP (CC3M) | `TripletCLIP/CC3M_TripletCLIP_ViTB12` |
| TripletCLIP (CC12M) | `TripletCLIP/CC12M_TripletCLIP_ViTB12` |
| NegCLIP (CC3M) | `TripletCLIP/CC3M_NegCLIP_ViTB12` |
| NegCLIP (CC12M) | `TripletCLIP/CC12M_NegCLIP_ViTB12` |
| LaCLIP, NegCLIP++, etc. | Same Hub page |

### Download example

Store lab checkpoints under this project when evaluating locally:

```bash
mkdir -p /mnt/data/TripletCLIP/checkpoints

uv run hf download TripletCLIP/CC12M_TripletCLIP_ViTB12 \
  --local-dir /mnt/data/TripletCLIP/checkpoints/CC12M_TripletCLIP_ViTB12
```

Batch download for main baselines:

```bash
for repo in \
  TripletCLIP/CC12M_TripletCLIP_ViTB12 \
  TripletCLIP/CC3M_TripletCLIP_ViTB12 \
  TripletCLIP/CC12M_NegCLIP_ViTB12
do
  name=$(basename "$repo")
  uv run hf download "$repo" --local-dir "/mnt/data/TripletCLIP/checkpoints/$name"
done
```

A separate Google Drive [OpenCLIP finetuning checkpoint](https://drive.google.com/file/d/14mupW26LMh6U4FQPa74FOIMEg8MndxCh/view) is documented in `README.md` for continued training on TripletData tars.

## Training (TripletData)

From `src/openclip/` after installing `requirements.txt`:

```bash
cd /mnt/data/TripletCLIP/src/openclip
python src/main.py \
  --model_name ViT-B-32 \
  --lr 0.00005 \
  --data_dir /path/to/tripletdata/tar/shards \
  --epochs 30 \
  --train \
  --log_dir /path/to/logs
```

TripletData training is configured through `main.py` CLI arguments (`--data_dir`, `--model_name`, `--lr`, `--epochs`, etc.). That path is separate from the lab pedestrian evaluation rules below.

## Config injection and dataset naming

Lab pedestrian evaluation uses Hugging Face checkpoints via `HubReIDCLIP` in `sugarcrepe-pedes.py`. Evaluation scripts must **not** pick a dataset or config file by default. Every run injects the YAML path through a required environment variable; if the variable is missing or empty, the script raises an error.

| Script | Environment variable | Config prefix |
|--------|---------------------|---------------|
| `sugarcrepe-pedes.py` | `SUGARCREPE_CONFIG` | `sugarcrepe` |
| `text-to-image-retrieval.py` | `RETRIEVAL_CONFIG` | `text_to_image_retrieval` |

Per-dataset config files use the suffix pattern `configs/{prefix}_{dataset_slug}.yaml`:

- `configs/sugarcrepe_cuhk_pedes.yaml` (`dataset: cuhk-pedes`)
- `configs/sugarcrepe_icfg_pedes.yaml` (`dataset: icfg-pedes`)
- `configs/sugarcrepe_rstpreid.yaml` (`dataset: rstpreid`)
- `configs/text_to_image_retrieval_cuhk_pedes.yaml` (same slug pattern for retrieval)

Each YAML must set `dataset` explicitly. Shell wrappers under `shell/` export the matching config paths before calling Python:

```bash
sh shell/eval_icfg_pedes.sh
```

Or manually:

```bash
export SUGARCREPE_CONFIG=configs/sugarcrepe_icfg_pedes.yaml
export RETRIEVAL_CONFIG=configs/text_to_image_retrieval_icfg_pedes.yaml
uv run python text-to-image-retrieval.py
uv run python sugarcrepe-pedes.py
```

CLI arguments are not supported for evaluation runs.

TripletData training (`main.py` CLI args, WebDataset shard paths) and lab pedestrian evaluation (env-injected YAML, explicit `dataset` field) are independent configuration surfaces. Do not mix CLI dataset selection into pedestrian eval wrappers.

## Lab compositional evaluation (pedestrian)

For in-lab SugarCrepe-style probes on CUHK-PEDES / ICFG-PEDES / RSTPReid, use Hugging Face pretrained weights with `HubReIDCLIP` (`checkpoints/CC12M_TripletCLIP_ViTB12` or other Hub downloads under `checkpoints/`). Hub models are trained on web data, not in-domain ReID. Pedestrian probe annotations come from `negative-pedestrians/outputs/{dataset}/` (see `env/.env` for `NEGATIVE_REID_DATASET_PATH`).
