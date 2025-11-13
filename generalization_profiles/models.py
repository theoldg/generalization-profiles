from typing import Self

from transformers import AutoTokenizer, GPTNeoXForCausalLM

MODEL_VARIANTS = {
    '70m-deduped',
}

# Valid revisions are 0, 1000, 2000, ..., 143000 (for all models).
VALID_REVISIONS = {str(step) for step in range(0, 144000, 1000)}


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
        revision: str,
    ) -> Self:
        if variant not in MODEL_VARIANTS:
            raise ValueError(f'Invalid model variant: {variant}')
        if revision not in VALID_REVISIONS:
            raise ValueError(
                f'Invalid revision: {revision}. '
                '(Should be a number from 0 to 143000 divisible by 1000).'
            )

        model = GPTNeoXForCausalLM.from_pretrained(
            'EleutherAI/pythia-70m-deduped',
            revision='step0',
            cache_dir='./pythia-70m-deduped/step3000',
        )

        tokenizer = AutoTokenizer.from_pretrained(
            'EleutherAI/pythia-70m-deduped',
            revision='step0',
            cache_dir='./pythia-70m-deduped/step3000',
        )

        return cls(model=model, tokenizer=tokenizer)


# inputs = tokenizer("Hello, I am", return_tensors="pt")
# tokens = model.generate(**inputs)
# output = tokenizer.decode(tokens[0])

# print(output)
