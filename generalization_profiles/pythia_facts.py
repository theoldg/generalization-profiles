# For the deduped models, the first step which started seeing repeated data.
# https://github.com/EleutherAI/pythia/issues/144
DEDUP_SECOND_EPOCH_START = 99_000

BATCH_SIZE = 1024

CHECKPOINT_INTERVAL = 1000

def seq_idx_to_batch(i: int) -> int:
    return i // BATCH_SIZE
