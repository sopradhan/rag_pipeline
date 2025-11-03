"""synthetic_data.py
YAML-driven synthetic data generator.

Schema example (see sample_schema.yaml):
  n: 200
  fields:
    timestamp:
      type: date
      start: '2020-01-01'
      freq: 'D'
    value:
      type: normal
      mean: 0
      std: 1
    wave:
      type: sin
      amplitude: 1.0
      period: 30
      noise: 0.1
    category:
      type: categorical
      categories: ['A','B','C']
      weights: [0.6, 0.3, 0.1]

Functions:
  - generate_from_schema(schema: dict) -> pandas.DataFrame
  - generate_from_yaml(path: str) -> pandas.DataFrame
"""
from typing import Dict, Any
import numpy as np
import pandas as pd

try:
    import yaml
except Exception:
    yaml = None


def generate_from_schema(schema: Dict[str, Any]):
    n = int(schema.get('n', 100))
    fields = schema.get('fields', {})
    data = {}

    # Helper index array
    idx = np.arange(n)

    for name, spec in fields.items():
        t = spec.get('type', 'normal')
        if t == 'date':
            start = spec.get('start', '2020-01-01')
            freq = spec.get('freq', 'D')
            # pandas date_range covers it
            data[name] = pd.date_range(start=start, periods=n, freq=freq)

        elif t == 'normal':
            mean = float(spec.get('mean', 0.0))
            std = float(spec.get('std', 1.0))
            data[name] = np.random.default_rng().normal(loc=mean, scale=std, size=n)

        elif t == 'uniform':
            low = float(spec.get('low', 0.0))
            high = float(spec.get('high', 1.0))
            data[name] = np.random.default_rng().uniform(low=low, high=high, size=n)

        elif t in ('sin', 'cos'):
            amp = float(spec.get('amplitude', 1.0))
            period = float(spec.get('period', 30.0))
            phase = float(spec.get('phase', 0.0))
            noise = float(spec.get('noise', 0.0))
            omega = 2 * np.pi / period
            base = amp * (np.sin(omega * idx + phase) if t == 'sin' else np.cos(omega * idx + phase))
            if noise > 0:
                base = base + np.random.default_rng().normal(scale=noise, size=n)
            data[name] = base

        elif t == 'categorical':
            cats = list(spec.get('categories', []))
            weights = spec.get('weights')
            if not cats:
                raise ValueError(f'categorical field {name} requires non-empty categories')
            if weights:
                weights = np.array(weights, dtype=float)
                weights = weights / weights.sum()
            data[name] = np.random.choice(cats, size=n, p=weights)

        elif t == 'id':
            start = int(spec.get('start', 0))
            step = int(spec.get('step', 1))
            data[name] = start + idx * step

        else:
            # fallback: fill with zeros
            data[name] = np.zeros(n)

    df = pd.DataFrame(data)
    return df


def generate_from_yaml(path: str):
    if yaml is None:
        raise RuntimeError('PyYAML is required for generate_from_yaml; please install PyYAML')
    with open(path, 'r', encoding='utf-8') as f:
        schema = yaml.safe_load(f)
    return generate_from_schema(schema)


if __name__ == '__main__':
    # quick smoke test if run directly
    sample = {
        'n': 100,
        'fields': {
            'ts': {'type': 'date', 'start': '2021-01-01', 'freq': 'D'},
            'value': {'type': 'normal', 'mean': 0, 'std': 1},
            'wave': {'type': 'sin', 'amplitude': 2.0, 'period': 30, 'noise': 0.1},
            'cat': {'type': 'categorical', 'categories': ['A', 'B'], 'weights': [0.7, 0.3]},
        }
    }
    df = generate_from_schema(sample)
    print('Generated sample shape:', df.shape)
    print(df.head())
