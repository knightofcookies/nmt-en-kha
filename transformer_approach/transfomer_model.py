import math
import re
import unicodedata
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_TOKEN = 0
EOS_TOKEN = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def read_langs():
    print("Reading lines...")

    with open("toy_datasets/kha_monolingual.txt", "r", encoding="utf-8") as fp:
        kha_monolingual_lines = fp.readlines()

    with open("toy_datasets/en_monolingual.txt", "r", encoding="utf-8") as fp:
        en_monolingual_lines = fp.readlines()

    with open("toy_datasets/kha_parallel.txt", "r", encoding="utf-8") as fp:
        kha_parallel_lines = fp.readlines()

    with open("toy_datasets/en_parallel.txt", "r", encoding="utf-8") as fp:
        en_parallel_lines = fp.readlines()

    assert len(kha_parallel_lines) == len(en_parallel_lines)

    kha_lang = Lang("khasi")
    en_lang = Lang("english")

    return (
        kha_lang,
        en_lang,
        kha_monolingual_lines,
        en_monolingual_lines,
        kha_parallel_lines,
        en_parallel_lines,
    )


def prepare_data():
    (
        kha_lang,
        en_lang,
        kha_monolingual_lines,
        en_monolingual_lines,
        kha_parallel_lines,
        en_parallel_lines,
    ) = read_langs()

    kha_monolingual_lines = [normalize_string(line) for line in kha_monolingual_lines]
    en_monolingual_lines = [normalize_string(line) for line in en_monolingual_lines]
    kha_parallel_lines = [normalize_string(line) for line in kha_parallel_lines]
    en_parallel_lines = [normalize_string(line) for line in en_parallel_lines]

    print("Counting words...")
    for kha_line in kha_monolingual_lines:
        kha_lang.add_sentence(normalize_string(kha_line))
    for en_line in en_monolingual_lines:
        en_lang.add_sentence(normalize_string(en_line))
    print("Counted words:")
    print(kha_lang.name, kha_lang.n_words)
    print(en_lang.name, en_lang.n_words)
    return (
        kha_lang,
        en_lang,
        kha_monolingual_lines,
        en_monolingual_lines,
        kha_parallel_lines,
        en_parallel_lines,
    )


(
    kha_lang,
    en_lang,
    kha_monolingual_lines,
    en_monolingual_lines,
    kha_parallel_lines,
    en_parallel_lines,
) = prepare_data()

# print(kha_lang.word2index)
# exit(1)


# Define the dataset class
class ParallelDataset(Dataset):
    def __init__(self, src_lines, tgt_lines, src_lang, tgt_lang):
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, index):
        src_line = self.src_lines[index].strip()
        tgt_line = self.tgt_lines[index].strip()

        src_indices = [self.src_lang.word2index[word] for word in src_line.split()] + [
            EOS_TOKEN
        ]
        tgt_indices = [self.tgt_lang.word2index[word] for word in tgt_line.split()] + [
            EOS_TOKEN
        ]

        return torch.tensor(src_indices), torch.tensor(tgt_indices)


class MonolingualDataset(Dataset):
    def __init__(self, lines, lang):
        self.lines = lines
        self.lang = lang

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index].strip()
        indices = [self.lang.word2index[word] for word in line.split()] + [EOS_TOKEN]
        return torch.tensor(indices)


# Create datasets and dataloaders
parallel_dataset = ParallelDataset(
    kha_parallel_lines, en_parallel_lines, kha_lang, en_lang
)
kha_monolingual_dataset = MonolingualDataset(kha_monolingual_lines, kha_lang)
en_monolingual_dataset = MonolingualDataset(en_monolingual_lines, en_lang)


# Helper function to pad sequences
def pad_sequences(batch):
    # Check if the batch contains pairs of sequences (parallel data)
    is_parallel = isinstance(batch[0], tuple)

    if is_parallel:
        # Sort the sequences by length (descending)
        sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

        # Separate source and target sequences
        src_seqs = [x[0] for x in sorted_batch]
        tgt_seqs = [x[1] for x in sorted_batch]

        # Pad the sequences
        src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=0)
        tgt_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=0)

        return src_padded, tgt_padded
    else:
        # Monolingual data
        sorted_batch = sorted(batch, key=len, reverse=True)
        padded = pad_sequence(sorted_batch, batch_first=True, padding_value=0)
        return padded


BATCH_SIZE = 32
parallel_dataloader = DataLoader(
    parallel_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_sequences
)
kha_monolingual_dataloader = DataLoader(
    kha_monolingual_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=pad_sequences,
)
en_monolingual_dataloader = DataLoader(
    en_monolingual_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=pad_sequences,
)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        print(token_embedding.shape)
        print(self.pos_embedding)
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size).to(DEVICE)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        hidden_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(hidden_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, hidden_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, dropout=dropout).to(
            device
        )

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


if __name__ == "main":

    HIDDEN_SIZE = 256

    kha_vocab_size = kha_lang.n_words
    en_vocab_size = en_lang.n_words

    kha_to_en_model = Seq2SeqTransformer(
        num_encoder_layers=3,
        num_decoder_layers=3,
        hidden_size=HIDDEN_SIZE,
        nhead=8,
        src_vocab_size=kha_vocab_size,
        tgt_vocab_size=en_vocab_size,
    )

    en_to_kha_model = Seq2SeqTransformer(
        num_encoder_layers=3,
        num_decoder_layers=3,
        hidden_size=HIDDEN_SIZE,
        nhead=8,
        src_vocab_size=en_vocab_size,
        tgt_vocab_size=kha_vocab_size,
    )

    kha_to_en_model = kha_to_en_model.to(DEVICE)
    en_to_kha_model = en_to_kha_model.to(DEVICE)

    # Initialize source token embeddings with pre-trained Word2Vec embeddings
    # kha_to_en_model.src_tok_emb.embedding.weight.data.copy_(khasi_embeddings)
    # en_to_kha_model.src_tok_emb.embedding.weight.data.copy_(english_embeddings)

    # Hyperparameters for the loss function
    LAMBDA_K = 1.0  # Weight for monolingual Khasi loss
    LAMBDA_E = 1.0  # Weight for monolingual English loss
    LAMBDA_P_K = 1.0  # Weight for parallel Khasi-to-English loss
    LAMBDA_P_E = 1.0  # Weight for parallel English-to-Khasi loss

    # Define optimizer
    optimizer = torch.optim.Adam(
        list(kha_to_en_model.parameters()) + list(en_to_kha_model.parameters()),
        lr=0.0001,
    )

    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Calculate loss for D_K (Khasi Monolingual)
        loss_K = 0
        for x_i in kha_monolingual_dataloader:
            x_i = x_i.to(DEVICE)  # Move input tensor to GPU

            # Create necessary masks and padding masks (example using kha_to_en_model)
            src_mask = kha_to_en_model.transformer.generate_square_subsequent_mask(
                len(x_i)
            ).to(DEVICE)
            tgt_mask = kha_to_en_model.transformer.generate_square_subsequent_mask(
                len(x_i)
            ).to(DEVICE)
            src_padding_mask = (x_i == 0).transpose(0, 1).to(DEVICE)
            tgt_padding_mask = (x_i == 0).transpose(0, 1).to(DEVICE)
            memory_key_padding_mask = src_padding_mask

            # Call kha_to_en_model to get the translated sequence
            translated_sequence = kha_to_en_model(
                x_i,
                x_i,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                memory_key_padding_mask,
            )

            # Create masks and padding masks for en_to_kha_model
            src_mask_en_to_kha = (
                en_to_kha_model.transformer.generate_square_subsequent_mask(
                    len(translated_sequence)
                ).to(DEVICE)
            )
            tgt_mask_en_to_kha = (
                en_to_kha_model.transformer.generate_square_subsequent_mask(
                    len(translated_sequence)
                ).to(DEVICE)
            )
            src_padding_mask_en_to_kha = (
                (translated_sequence == 0).transpose(0, 1).to(DEVICE)
            )
            tgt_padding_mask_en_to_kha = (
                (translated_sequence == 0).transpose(0, 1).to(DEVICE)
            )
            memory_key_padding_mask_en_to_kha = src_padding_mask_en_to_kha

            # Call en_to_kha_model with the translated sequence and appropriate masks
            x_i_hat = en_to_kha_model(
                translated_sequence,
                translated_sequence,
                src_mask_en_to_kha,
                tgt_mask_en_to_kha,
                src_padding_mask_en_to_kha,
                tgt_padding_mask_en_to_kha,
                memory_key_padding_mask_en_to_kha,
            )

            loss_K += torch.norm(x_i - x_i_hat) ** 2
        loss_K = LAMBDA_K * loss_K / len(kha_monolingual_dataset)

        # Calculate loss for D_E (English Monolingual)
        loss_E = 0
        for x_i in en_monolingual_dataloader:
            x_i = x_i.to(DEVICE)  # Move input tensor to GPU

            tgt_input = x_i[:-1, :]


            # Create necessary masks and padding masks (example using en_to_kha_model)
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            memory_key_padding_mask = src_padding_mask

            # Call en_to_kha_model to get the translated sequence
            translated_sequence = en_to_kha_model(
                x_i,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                memory_key_padding_mask,
            )

            # Create masks and padding masks for en_to_kha_model
            src_mask_kha_to_en = (
                kha_to_en_model.transformer.generate_square_subsequent_mask(
                    len(translated_sequence)
                ).to(DEVICE)
            )
            tgt_mask_kha_to_en = (
                kha_to_en_model.transformer.generate_square_subsequent_mask(
                    len(translated_sequence)
                ).to(DEVICE)
            )
            src_padding_mask_kha_to_en = (
                (translated_sequence == 0).transpose(0, 1).to(DEVICE)
            )
            tgt_padding_mask_kha_to_en = (
                (translated_sequence == 0).transpose(0, 1).to(DEVICE)
            )
            memory_key_padding_mask_kha_to_en = src_padding_mask_kha_to_en

            # Call kha_to_en_model with the translated sequence and appropriate masks
            x_i_hat = kha_to_en_model(
                translated_sequence,
                translated_sequence,
                src_mask_kha_to_en,
                tgt_mask_kha_to_en,
                src_padding_mask_kha_to_en,
                tgt_padding_mask_kha_to_en,
                memory_key_padding_mask_kha_to_en,
            )

            loss_E += torch.norm(x_i - x_i_hat) ** 2
        loss_E = LAMBDA_K * loss_E / len(kha_monolingual_dataset)

        # Calculate loss for D_P (Parallel Data)
        loss_P_K = 0
        loss_P_E = 0
        for u_k, v_k in parallel_dataloader:
            u_k = u_k.to(DEVICE)  # Move data to the appropriate device
            v_k = v_k.to(DEVICE)  # Move data to the appropriate device
            v_k_hat = kha_to_en_model(u_k)
            u_k_hat = en_to_kha_model(v_k)
            loss_P_K += torch.norm(v_k - v_k_hat) ** 2
            loss_P_E += torch.norm(u_k - u_k_hat) ** 2
        loss_P_K = LAMBDA_P_K * loss_P_K / len(parallel_dataset)
        loss_P_E = LAMBDA_P_E * loss_P_E / len(parallel_dataset)

        # Total loss
        loss = loss_K + loss_E + loss_P_K + loss_P_E

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print loss for monitoring
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Save the trained models
    torch.save(kha_to_en_model.state_dict(), "kha_to_en_model.pth")
    torch.save(en_to_kha_model.state_dict(), "en_to_kha_model.pth")
