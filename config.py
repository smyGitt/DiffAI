from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 512
    emb_dim: int = 256
    n_heads: int = 8
    n_layers: int = 6
    drop_rate: float = 0.1
    qkv_bias: bool = False


@dataclass
class DataConfig:
    csv_path: Path = None
    max_diff_tokens: int = 400
    max_msg_tokens: int = 80
    max_rows_train: int = 50_000
    max_rows_val: int = 5_000
    max_rows_test: int = 5_000


@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 3
    lr: float = 3e-4
    eval_every: int = 500
    save_every: int = 2000
    checkpoint_dir: Path = Path("checkpoints")
    use_pretrained: bool = True
    pretrained_model: str = "gpt2"
    freeze_backbone: bool = False
    unfreeze_last_n: int = 4
    blind_rate: float = 0.3
