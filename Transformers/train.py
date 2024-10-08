from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from Transformers.util import get_dataset
from Transformers.Layers import greedy_decode
from Transformers.util import get_model
from config import get_weights_file_path, get_config

from torch.utils.tensorboard.writer import SummaryWriter

def run_validation(
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_state,
    writer,
    num_example,
):
    model.eval()
    count = 0

    source_texts = []
    target = []
    predicted = []

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batchsize 1 for val"

            out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )
            print(out.shape)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]

            output_text = tokenizer_tgt.decode(out.detach().cpu().numpy())
            source_texts.append(source_text)
            target.append(target_text)
            predicted.append(output_text)

            print_msg(f"\n source: {source_text}")
            print_msg(f"\n target: {target_text}")
            print_msg(f"\n pred: {output_text}")

            if count == num_example:
                break


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)

    model = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    ).to(device)

    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    init_epoch = 0
    global_step = 0

    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"loading model from {model_filename}")
        state = torch.load(model_filename)
        init_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    for epoch in range(init_epoch, config["num_epoch"]):
        model.train(True)

        batch_iterator = tqdm(train_dataloader, desc=f"Processing epooch {epoch:02d}")

        for batch in batch_iterator:

            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)  # (b, s, d)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (b, s, d)

            # (b, s, vocabsize)
            proj = model.project(decoder_output)

            label = batch["label"].to(device)

            # b* s , vocab size for cross ent
            loss = loss_fn(
                proj.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            run_validation(
                model,
                val_dataloader,
                tokenizer_src,
                tokenizer_tgt,
                config["seq_len"],
                device,
                lambda msg: batch_iterator.write(msg),
                global_step,
                writer,
                5,
            )

            global_step += 1

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epooch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    config = get_config()
    train_model(config)
