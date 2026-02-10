from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from collections import deque
import time

from tqdm.auto import tqdm
from fire import Fire
import pandas as pd
from google import genai
from google.genai import types
import numpy as np


class EmbeddingClient:
    def __init__(
        self, 
        model: str = 'gemini-embedding-001',
        embedding_size: int = 768,  # 768, 1536, or 3072 
    ):
        self._client = genai.Client()
        self._model = model
        self._config = types.EmbedContentConfig(
            output_dimensionality=embedding_size
        )

    def __call__(self, strings: list[str]) -> np.ndarray:
        response = self._client.models.embed_content(
            model=self._model,
            contents=strings,  # type: ignore
            config=self._config,
        )
        assert response.embeddings is not None
        ret = np.vstack([
            np.array(e.values) for e in response.embeddings
        ])
        ret /= np.linalg.norm(ret, axis=1)[:, None]
        return ret


@dataclass
class RateLimiter:
    max_calls: int
    interval: timedelta

    def __post_init__(self):
        self._timestamps = deque()

    def wait(self):
        now = time.monotonic()
        window = self.interval.total_seconds()

        while self._timestamps and (now - self._timestamps[0]) > window:
            self._timestamps.popleft()

        if len(self._timestamps) >= self.max_calls:
            sleep_time = self._timestamps[0] + window - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._timestamps.popleft()

        self._timestamps.append(time.monotonic())
        

def save_batch(
    data: list[np.ndarray],
    index: int,
    target_path: Path,
):
    target_path.mkdir(exist_ok=True, parents=True)
    x = np.vstack(data)
    np.save(target_path / str(index), x)


def combine_batches(
    batches_path: Path,
    target_path: str,
):
    def parse_batch_number(p: Path):
        return int(p.with_suffix('').name)

    paths = sorted(
        batches_path.glob('*.npy'),
        key=parse_batch_number,
    )

    with ThreadPoolExecutor(32) as executor:
        loaded_batches = list(executor.map(
            np.load, paths
        ))
    
    combined = np.vstack(loaded_batches)
    np.save(target_path, combined)


def main(
    segments_path: str = 'results/segments.parquet',
    target_path: str = 'results/embeddings/batches',  # type: ignore
    rpm: int = 2900,
    n_threads: int = 32,
    batch_size: int = 2000,
):
    segments_df = pd.read_parquet(segments_path)
    client = EmbeddingClient()

    target_path: Path = Path(target_path)
    starting_batch = 0
    while (target_path / f'{starting_batch}.npy').exists():
        starting_batch += 1

    def iter_text_at_rate_limit():
        limiter = RateLimiter(
            max_calls=rpm,
            interval=timedelta(minutes=1),
        )
        for s in tqdm(segments_df.text[starting_batch * batch_size:]):
            limiter.wait()
            yield s

    def iter_embeddings():
        with ThreadPoolExecutor(n_threads) as executor:
            text_it = iter_text_at_rate_limit()
            tasks = deque()
            for t, _ in zip(text_it, range(n_threads * 2)):
                tasks.append(executor.submit(client, [t]))

            for t in text_it:
                yield tasks.popleft().result()
                tasks.append(executor.submit(client, [t]))
            
            while tasks:
                yield tasks.popleft().result()
            
    batch = []
    i = starting_batch
    for e in iter_embeddings():
        batch.append(e)
        if len(batch) == batch_size:
            save_batch(batch, i, target_path=target_path)
            i += 1
            batch = []

    if batch:
        save_batch(batch, i, target_path=target_path)

    combine_batches(
        batches_path=target_path,
        target_path='results/embeddings/segment_embeddings.npy',
    )

if __name__ == '__main__':
    Fire(main)
