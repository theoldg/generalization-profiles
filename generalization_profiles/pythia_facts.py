MODEL_VARIANTS: list[str] = [
    "70m",
    "160m",
    "410m",
    "1.4b",
    "2.8b",
    "6.9b",
    "12b",
]

# Valid revisions are 0, 1000, 2000, ..., 143000 for all models.
VALID_REVISIONS: set[int] = {step for step in range(0, 144000, 1000)}


# For the deduped models, the first step which started seeing repeated data.
# https://github.com/EleutherAI/pythia/issues/144
DEDUP_SECOND_EPOCH_START = 99_000

BATCH_SIZE = 1024

# Sequences with seq_idx equal or greater than this should be considered already seen.
FIRST_REPEATED_SEQ_IDX = DEDUP_SECOND_EPOCH_START * BATCH_SIZE

CHECKPOINT_INTERVAL = 1000
