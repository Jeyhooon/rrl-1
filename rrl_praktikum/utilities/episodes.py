import datetime
import io
import pathlib
import uuid

import numpy as np


def count_episodes(directory):
    filenames = directory.glob('*.npz')
    lengths = [int(n.stem.rsplit('-', 1)[-1]) - 1 for n in filenames]
    episodes, steps = len(lengths), sum(lengths)
    return episodes, steps


def save_episodes(directory, episodes):
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    for episode in episodes:
        identifier = str(uuid.uuid4().hex)
        length = len(episode['reward'])
        filename = directory / f'{timestamp}-{identifier}-{length}.npz'
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open('wb') as f2:
                f2.write(f1.read())


def load_episodes(directory, rescan, length=None, balance=False, seed=0):
    directory = pathlib.Path(directory).expanduser()
    random = np.random.RandomState(seed)
    cache = {}
    while True:
        for filename in directory.glob('*.npz'):
            if filename not in cache:
                try:
                    with filename.open('rb') as f:
                        episode = np.load(f)
                        episode = {k: episode[k] for k in episode.keys()}
                except Exception as e:
                    print(f'Could not load episode: {e}')
                    continue
                cache[filename] = episode
        keys = list(cache.keys())
        for index in random.choice(len(keys), rescan):
            episode = cache[keys[index]]
            if length:
                total = len(next(iter(episode.values())))
                available = total - length
                if available < 1:
                    print(f'Skipped short episode of length {available}.')
                    continue
                if balance:
                    index = min(random.randint(0, total), available)
                else:
                    index = int(random.randint(0, available))
                episode = {k: v[index: index + length] for k, v in episode.items()}
            yield episode
