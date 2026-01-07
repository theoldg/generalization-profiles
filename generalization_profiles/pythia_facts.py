# For the deduped models, the first step which started seeing repeated data.
# https://github.com/EleutherAI/pythia/issues/144
DEDUP_SECOND_EPOCH_START = 99_000

BATCH_SIZE = 1024

# Sequences with seq_idx equal or greater than this should be considered already seen.
FIRST_REPEATED_SEQ_IDX = DEDUP_SECOND_EPOCH_START * BATCH_SIZE

CHECKPOINT_INTERVAL = 1000

def seq_idx_to_batch(i: int) -> int:
    return i // BATCH_SIZE
