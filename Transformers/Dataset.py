from Transformers.Layers import causal_mask


import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset



class BiLingualDataSet(Dataset):

    def __init__(
        self,
        ds,
        tokenizer_src: Tokenizer,
        tokenizer_tgt: Tokenizer,
        src_lang,
        tgt_lang,
        seq_len,
    ) -> None:
        super().__init__()
        self.ds = ds

        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor(
            [tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_src.encode(tgt_text).ids

        padding_enc = self.seq_len - len(enc_input_tokens) - 2
        padding_dec = self.seq_len - len(dec_input_tokens) - 1

        assert (
            padding_enc >= 0 and padding_dec >= 0
        ), "Seq len less than max len in dataset"

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * padding_enc),
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * padding_dec),
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * padding_dec),
            ]
        )

        assert (
            encoder_input.size(0) == self.seq_len
            and decoder_input.size(0) == self.seq_len
            and label.size(0) == self.seq_len
        ), "One of the data seq len != seq_len"

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0),  # (1,1 seq len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0)
            & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


