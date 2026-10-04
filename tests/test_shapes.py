"""Shape and sanity checks for the model (no data download needed).

    python -m tests.test_shapes
"""
import torch
import torch.nn as nn

from src.config import Config
from src.data import Vocab
from src.model import build_model, count_parameters

B, S, T = 4, 7, 9          # batch, source length, target length (incl. <SOS>/<EOS>)


def make_vocabs():
    en = Vocab(["a b c d e f g h"] * 2, min_freq=1)        # 4 specials + 8 words
    hi = Vocab(["p q r s t u v w x y"] * 2, min_freq=1)    # 4 specials + 10 words
    return en, hi


def expected_rnn_params(cell_type, E, H):
    gates = {"rnn": 1, "gru": 3, "lstm": 4}[cell_type]
    return gates * (E * H + H * H + 2 * H)


def check_cell(cell_type):
    torch.manual_seed(0)
    cfg = Config(cell_type=cell_type, embedding_dim=16, hidden_dim=32)
    en, hi = make_vocabs()
    model = build_model(cfg, en, hi)
    E, H, L = cfg.embedding_dim, cfg.hidden_dim, cfg.num_layers

    # Batch with different real lengths; padding at the end
    lengths = torch.tensor([7, 5, 3, 6])
    src = torch.randint(4, len(en), (B, S))
    for i, n in enumerate(lengths):
        src[i, n:] = en.pad_idx
    tgt = torch.randint(4, len(hi), (B, T))
    tgt[:, 0] = hi.sos_idx
    tgt[:, -1] = hi.eos_idx

    # ---- Encoder ----
    outputs, state = model.encoder(src, lengths)
    assert outputs.shape == (B, S, H), outputs.shape
    h = state[0] if cell_type == "lstm" else state
    assert h.shape == (L, B, H), h.shape
    if cell_type == "lstm":
        assert state[1].shape == (L, B, H)

    # With packing, the final hidden state equals the output at each
    # sentence's LAST REAL step (not at the last padded position)
    for i, n in enumerate(lengths):
        assert torch.allclose(h[-1, i], outputs[i, n - 1], atol=1e-6)
        # Padded positions of the outputs are zeros
        assert torch.all(outputs[i, n:] == 0)

    # ---- One decoder step ----
    logits, _ = model.decoder(tgt[:, 0], state)
    assert logits.shape == (B, len(hi)), logits.shape

    # ---- Full forward (teacher forcing) + loss ----
    out = model(src, lengths, tgt, teacher_forcing_ratio=1.0)
    assert out.shape == (B, T - 1, len(hi)), out.shape
    loss = nn.CrossEntropyLoss(ignore_index=hi.pad_idx)(
        out.reshape(-1, len(hi)), tgt[:, 1:].reshape(-1)
    )
    loss.backward()
    assert torch.isfinite(loss)

    # <PAD> embedding rows receive no gradient
    assert torch.all(model.encoder.embedding.weight.grad[en.pad_idx] == 0)

    # ---- Greedy decoding ----
    preds = model.greedy_decode(src, lengths, max_len=12)
    assert preds.shape[0] == B and preds.shape[1] <= 12

    # ---- Parameter count by hand ----
    by_hand = (
        len(en) * E                                  # encoder embedding
        + expected_rnn_params(cell_type, E, H)       # encoder RNN
        + len(hi) * E                                # decoder embedding
        + expected_rnn_params(cell_type, E, H)       # decoder RNN
        + H * len(hi) + len(hi)                      # output layer
    )
    assert count_parameters(model) == by_hand, (count_parameters(model), by_hand)
    print(f"{cell_type:4s} OK | params {by_hand:,} | loss {loss.item():.3f}")


if __name__ == "__main__":
    for cell in ["rnn", "gru", "lstm"]:
        check_cell(cell)
    print("All shape tests passed.")
