import typing

import datasets
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import pandas as pd


@typing.no_type_check
def load_dataset(tokenizer: AutoTokenizer) -> pd.DataFrame:
    data = datasets.load_dataset("pietrolesci/pile-deduped-subset")
    all_data = datasets.concatenate_datasets([data["train"], data["validation"]])
    df: pd.DataFrame = all_data.with_format("pandas")[:]

    assert len(set(all_data["seq_idx"])) == len(all_data)

    tokens = np.stack(df.input_ids.values)
    df["text"] = tokenizer.batch_decode(tokens)
    return df


def load_spacy_model():
    return spacy.load(
        "en_core_web_trf",
        disable=["ner", "tagger", "attribute_ruler", "lemmatizer"],
    )


def mask_list(lst: list, mask: np.ndarray) -> list:
    # Do the same as x[boolean_index] for numpy, except with a list.
    assert mask.ndim == 1
    assert len(lst) == len(mask)
    return [lst[i] for i in np.where(mask)[0]]


def segment_tokens(
    seq: np.ndarray,
    segments: list[str],
    tokenizer: AutoTokenizer,
    minimum_tokens_per_segment: int = 2,
) -> pd.DataFrame:
    """
    Breaks the text into sentences at the string level, and tries to locate
    those sentences in the token sequence.

    Because of whitespace differences at the start and end, the matching
    ignores the first and last token of each sequence.

    Only a subset of the sentences end up included because we require:
      - A minimum number of tokens per sentence (2 by default),
      - Exactly one match at the token level: repeated sentences are dropped
        and matching errors are quietly skipped.

    Args:
        seq: The token sequence (1D array) to search within.
        text: The raw string corresponding to `seq`.
        tokenizer: The HuggingFace tokenizer used to process segments.
        minimum_tokens_per_segment: Minimum length a segment must have to be included.
    """
    tokenization_result = tokenizer(
        segments,
        padding=True,
        return_tensors="np",
        add_special_tokens=False,
    )  # type: ignore
    seg_ids = tokenization_result["input_ids"]
    mask = tokenization_result["attention_mask"].astype(bool)

    seg_ids[~mask] = -1

    # Mask the first and last token (whitespace mismatches etc)
    seg_ids[:, 0] = -1
    seg_ids[np.arange(len(seg_ids)), mask.sum(-1) - 1] = -1

    # seq_strided[i] is [seq[i], seq[i+1], ..., seq[i + max_length - 1]]
    _, max_length = seg_ids.shape
    seq_strided = np.lib.stride_tricks.sliding_window_view(
        np.concatenate((seq, -np.ones(max_length - 1, dtype=seq.dtype))),
        window_shape=max_length,
    )

    # Compare everything to everything
    x = seq_strided[:, None, :] == seg_ids[None, :, :]
    x |= seg_ids[None, :, :] == -1
    # matches[i, j] is 1 iff segment j is found at starting position i in the original seq
    matches = x.all(-1)

    num_matches = matches.sum(0)
    start_idx = matches.argmax(0)

    # Drop segments with too few tokens
    tokens_per_segment = mask.sum(-1)
    valid_seg_mask = tokens_per_segment >= minimum_tokens_per_segment
    # Only accept segments with exactly one match
    valid_seg_mask &= num_matches == 1

    return pd.DataFrame(
        {
            "start_idx": start_idx[valid_seg_mask],
            "text": mask_list(segments, valid_seg_mask),
            "num_tokens": tokens_per_segment[valid_seg_mask],
        }
    )


def segment_dataset_to_parquet(
    target_path='results/segmentation_results.parquet',
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/pythia-70m",
        revision=f"step0",
    )
    tokenizer.pad_token = '[PAD]'  # idk
    df = load_dataset(tokenizer)

    result = []

    spacy_model = load_spacy_model()
    for (_, r), doc in tqdm(
        zip(
            df.iterrows(),
            spacy_model.pipe(
                df.text,
                batch_size=8,
                n_process=1,
            ),
        ),
        total=len(df),
    ):
        segments = [str(s) for s in doc.sents]
        seg = segment_tokens(
            seq=r.input_ids,
            segments=segments,
            tokenizer=tokenizer,
        )
        seg["seq_idx"] = r.seq_idx
        result.append(seg)

    result = pd.concat(result)
    result.to_parquet(target_path)
