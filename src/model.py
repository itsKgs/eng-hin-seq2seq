"""Encoder, Decoder and the full Seq2Seq model (no attention).

Shapes (B = batch, S = src_len, T = tgt_len, E = embedding_dim,
        H = hidden_dim, L = num_layers, V = vocab size):

    src            [B, S]
    embedded src   [B, S, E]
    encoder out    [B, S, H]
    hidden         [L, B, H]          (and cell [L, B, H] for LSTM)
    decoder step:  input [B] -> logits [B, V_tgt]
    Seq2Seq out    [B, T-1, V_tgt]
"""
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Same interface for all three: switching cell_type changes only this lookup
RNN_CLASSES = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}


class Encoder(nn.Module):
    """English ids -> per-step outputs + final state (h, or (h, c) for LSTM)."""

    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers,
                 cell_type, pad_idx, use_packing=True):
        super().__init__()
        self.use_packing = use_packing

        # padding_idx: the <PAD> row stays zero and never receives gradient
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = RNN_CLASSES[cell_type](
            emb_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )

    def forward(self, src, src_lengths):
        # src: [B, S]  ->  embedded: [B, S, E]
        embedded = self.embedding(src)

        if self.use_packing:
            # Pack so the RNN skips <PAD> steps: the final state is the state
            # after each sentence's LAST REAL word. Lengths must be on the CPU.
            packed = pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_outputs, state = self.rnn(packed)
            outputs, _ = pad_packed_sequence(
                packed_outputs, batch_first=True, total_length=src.size(1)
            )
        else:
            # Without packing the RNN also runs over <PAD> steps, so the final
            # state is "after the padding", not after the last real word.
            outputs, state = self.rnn(embedded)

        # outputs: [B, S, H]; state: h [L, B, H]  or  (h, c) for LSTM
        return outputs, state


class Decoder(nn.Module):
    """One decoding step: previous token + state -> vocabulary logits + new state."""

    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, cell_type, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = RNN_CLASSES[cell_type](
            emb_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        # z_t = W h_t + b  -> raw logits (softmax is applied inside the loss)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, state):
        # input_token: [B]  ->  [B, 1] (a sequence of length 1)
        embedded = self.embedding(input_token.unsqueeze(1))   # [B, 1, E]

        # state is h [L, B, H] (RNN/GRU) or (h, c) (LSTM); passed through as-is
        output, state = self.rnn(embedded, state)              # output: [B, 1, H]

        logits = self.fc_out(output.squeeze(1))                # [B, V_tgt]
        return logits, state


class Seq2Seq(nn.Module):
    """Encoder -> final state -> Decoder (autoregressive)."""

    def __init__(self, encoder, decoder, sos_idx, eos_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio):
        """
        Training / validation pass.

        tgt is the FULL target: <SOS> w1 ... wn <EOS>   [B, T]
          decoder inputs = tgt[:, :-1]  (start with <SOS>)
          labels         = tgt[:, 1:]   (end with <EOS>)  -> used in the loss
        Returns logits [B, T-1, V_tgt], aligned with tgt[:, 1:].
        """
        T = tgt.size(1)

        # The ONLY link between encoder and decoder: the final state
        _, state = self.encoder(src, src_lengths)

        input_token = tgt[:, 0]          # <SOS> for every sentence, [B]
        step_logits = []

        for t in range(1, T):
            logits, state = self.decoder(input_token, state)   # [B, V]
            step_logits.append(logits)

            # Teacher forcing: feed the TRUE previous word, otherwise the
            # model's own prediction. Skip the coin flip at ratio 0 or 1.
            if teacher_forcing_ratio >= 1.0:
                use_teacher = True
            elif teacher_forcing_ratio <= 0.0:
                use_teacher = False
            else:
                use_teacher = random.random() < teacher_forcing_ratio

            input_token = tgt[:, t] if use_teacher else logits.argmax(dim=1)

        return torch.stack(step_logits, dim=1)                # [B, T-1, V]

    @torch.no_grad()
    def greedy_decode(self, src, src_lengths, max_len):
        """
        Inference: no target available. Start from <SOS>, feed back the argmax,
        stop when every sentence has produced <EOS> or after max_len steps.
        Returns predicted ids [B, <= max_len].
        """
        B = src.size(0)
        _, state = self.encoder(src, src_lengths)

        # Created on the same device as the input (works on CPU and GPU)
        input_token = torch.full((B,), self.sos_idx, dtype=torch.long, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        predictions = []

        for _ in range(max_len):
            logits, state = self.decoder(input_token, state)
            pred = logits.argmax(dim=1)                       # [B]
            predictions.append(pred)

            finished |= pred == self.eos_idx
            if finished.all():
                break
            input_token = pred

        return torch.stack(predictions, dim=1)


def build_model(cfg, src_vocab, tgt_vocab, use_packing=True):
    """Create a Seq2Seq model from the config and the two vocabularies."""
    encoder = Encoder(
        vocab_size=len(src_vocab),
        emb_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        cell_type=cfg.cell_type,
        pad_idx=src_vocab.pad_idx,
        use_packing=use_packing,
    )
    decoder = Decoder(
        vocab_size=len(tgt_vocab),
        emb_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        cell_type=cfg.cell_type,
        pad_idx=tgt_vocab.pad_idx,
    )
    return Seq2Seq(encoder, decoder, tgt_vocab.sos_idx, tgt_vocab.eos_idx)


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
