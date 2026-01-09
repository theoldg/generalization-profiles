import datasets
from tqdm.auto import tqdm

from generalization_profiles.pythia import MODEL_VARIANTS


APPROX_DS_SIZE = 2_325_806  # 16k samples * 144 checkpoints - ish


def stream_dataset(model):
    ds = datasets.load_dataset(
        "pietrolesci/pythia-deduped-stats-raw",
        data_dir=model,
        streaming=True,
    )
    return ds['train']


# Ok i can stream over a single one in about 30 mins
# how about DDOS
for _ in tqdm(stream_dataset('6.9b/pythia-deduped-6.9b-step131000*'), total=APPROX_DS_SIZE):
    pass

## might come in handy
# from huggingface_hub import HfFileSystem
# fs = HfFileSystem()
# fs.ls("datasets/pietrolesci/pythia-deduped-stats-raw")
# for x in fs.walk("datasets/pietrolesci/pythia-deduped-stats-raw"):
#     print(x)