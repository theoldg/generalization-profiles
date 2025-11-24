from pathlib import Path
from typing import Self

from transformers import AutoTokenizer, GPTNeoXForCausalLM

MODEL_VARIANTS: set[str] = {
    '70m-deduped',
}

# Valid revisions are 0, 1000, 2000, ..., 143000 for all models.
VALID_REVISIONS: set[int] = {step for step in range(0, 144000, 1000)}


class PythiaModel:
    model: GPTNeoXForCausalLM
    tokenizer: AutoTokenizer

    def __init__(self, model: GPTNeoXForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_variant_and_revision(
        cls,
        variant: str,
        revision: int,
        cache_dir: str | Path = 'cache/pythia',
    ) -> Self:
        cache_dir = Path(cache_dir)
        if variant not in MODEL_VARIANTS:
            raise ValueError(f'Invalid model variant: {variant}')
        if revision not in VALID_REVISIONS:
            raise ValueError(
                f'Invalid revision: {revision}. '
                '(Should be a number from 0 to 143000 divisible by 1000).'
            )
        model = GPTNeoXForCausalLM.from_pretrained(
            f'EleutherAI/pythia-{variant}',
            revision=f'step{revision}',
            cache_dir=cache_dir / variant / str(revision),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            f'EleutherAI/pythia-{variant}',
            revision=f'step{revision}',
            cache_dir=cache_dir / variant / str(revision),
        )
        # See huggingface.co/EleutherAI/pythia-6.9b/raw/main/tokenizer.json
        tokenizer.pad_token = '<|padding|>'
        return cls(model=model, tokenizer=tokenizer)
