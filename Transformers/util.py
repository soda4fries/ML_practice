from datasets import load_dataset
from torch.utils.data import DataLoader
from Transformers.Dataset import BiLingualDataSet, get_all_sentences
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from pathlib import Path
from Transformers.Layers import build_transformer
from Transformers.train import random_split


def get_model(config, vocab_src_len, vocab_tgt_len):

    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    )
    return model


def get_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_dataset(config):
    # load using datasets library
    ds_raw = load_dataset(
        "opus_books", f'{config["lang_src"]}-{config["lang_tgt"]}', split="train"
    )

    tokenizer_src = get_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_tokenizer(config, ds_raw, config["lang_tgt"])

    train_ds_size = int(0.8 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    raw_train_ds, raw_val_ds = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_dataset = BiLingualDataSet(
        raw_train_ds,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_dataset = BiLingualDataSet(
        raw_val_ds,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt = tokenizer_src.encode(item["translation"][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src))
        max_len_tgt = max(max_len_tgt, len(tgt))

    print(f"max src len: {max_len_src} \nmax tgt len: {max_len_tgt}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
