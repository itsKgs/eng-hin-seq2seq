from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42

    batch_size: int = 32

    # embedding_dim: vocabularies here are only a few thousand words
    # (V_en ~2.5k, V_hi ~3k), far smaller than the 100k+ vocabularies that
    # use 300-1000 dims. 256 keeps embeddings at ~0.64M (en) + ~0.77M (hi)
    # params, which is still generous for this data.
    embedding_dim: int = 256

    # hidden_dim: one PyTorch GRU = 3*(E*H + H*H + 2H) = 1,182,720 params
    # (two bias vectors per gate). Encoder + decoder GRUs ~2.37M, output
    # layer H*V_hi + V_hi ~1.54M -> total ~5.3M params vs ~2.5k training
    # pairs (~15k target tokens). The model can memorize the training set,
    # so expect overfitting: watch the train/val gap and keep the best-val
    # checkpoint.
    hidden_dim: int = 512

    num_layers: int = 1
    learning_rate: float = 1e-3
    num_epochs: int = 20
    teacher_forcing_ratio: float = 0.5
    clip_max_norm: float = 1.0
    max_decode_len: int = 50  # placeholder; derived from data in Step 4
    cell_type: str = "gru"
    data_dir: str = "data"

    def __post_init__(self):
        if self.cell_type not in {"rnn", "gru", "lstm"}:
            raise ValueError(
                f"Invalid cell_type: {self.cell_type!r}. "
                "Expected one of: 'rnn', 'gru', 'lstm'."
            )
        if not 0.0 <= self.teacher_forcing_ratio <= 1.0:
            raise ValueError(
                f"teacher_forcing_ratio must be in [0, 1], "
                f"got {self.teacher_forcing_ratio}"
            )
