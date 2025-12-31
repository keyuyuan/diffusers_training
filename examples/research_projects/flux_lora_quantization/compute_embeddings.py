#!/usr/bin/env python
# coding=utf-8

import argparse
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub.utils import insecure_hashlib
from tqdm.auto import tqdm
from transformers import T5EncoderModel
from diffusers import FluxPipeline

MAX_SEQ_LENGTH = 77
OUTPUT_PATH = "embeddings.parquet"


def generate_image_hash(image):
    # image is PIL.Image
    return insecure_hashlib.sha256(image.tobytes()).hexdigest()


def load_flux_dev_pipeline():
    model_id = "black-forest-labs/FLUX.1-dev"

    text_encoder = T5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder_2",
        load_in_8bit=True,
        device_map="auto",
    )

    pipeline = FluxPipeline.from_pretrained(
        model_id,
        text_encoder_2=text_encoder,
        transformer=None,
        vae=None,
        device_map="balanced",
    )
    return pipeline


@torch.no_grad()
def compute_embeddings(pipeline, prompts, max_sequence_length):
    prompt_embeds_all = []
    pooled_prompt_embeds_all = []
    text_ids_all = []

    for prompt in tqdm(prompts, desc="Encoding prompts"):
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            max_sequence_length=max_sequence_length,
        )
        prompt_embeds_all.append(prompt_embeds)
        pooled_prompt_embeds_all.append(pooled_prompt_embeds)
        text_ids_all.append(text_ids)

    max_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Max CUDA memory used: {max_mem:.2f} GB")

    return prompt_embeds_all, pooled_prompt_embeds_all, text_ids_all


def run(args):
    dataset = load_dataset(args.dataset_name, split="train")

    image_hashes = []
    image_paths = []
    prompts = []

    for sample in dataset:
        image = sample["image"]           # PIL.Image
        image_hashes.append(generate_image_hash(image))

        # ✅ 关键修复点：必须用 .path
        image_paths.append(image.path)

        prompts.append(sample[args.caption_column])

    print(f"Loaded {len(prompts)} samples")

    pipeline = load_flux_dev_pipeline()
    prompt_embeds, pooled_prompt_embeds, text_ids = compute_embeddings(
        pipeline, prompts, args.max_sequence_length
    )

    rows = []
    for i in range(len(prompts)):
        rows.append(
            (
                image_hashes[i],
                image_paths[i],
                prompt_embeds[i],
                pooled_prompt_embeds[i],
                text_ids[i],
            )
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "image_hash",
            "image_path",
            "prompt_embeds",
            "pooled_prompt_embeds",
            "text_ids",
        ],
    )

    # parquet 只能存 python 原生 / numpy
    df["prompt_embeds"] = df["prompt_embeds"].apply(lambda x: x.cpu().numpy().flatten().tolist())
    df["pooled_prompt_embeds"] = df["pooled_prompt_embeds"].apply(lambda x: x.cpu().numpy().flatten().tolist())
    df["text_ids"] = df["text_ids"].apply(lambda x: x.cpu().numpy().flatten().tolist())

    df.to_parquet(args.output_path)
    print(f"Saved embeddings to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--caption_column", type=str, default="text")
    parser.add_argument("--max_sequence_length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)

    args = parser.parse_args()
    run(args)
