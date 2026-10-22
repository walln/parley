# Parley

A full-duplex speech model: Gemma 4 12B listens while it talks and decides every 80 ms
whether to speak, keep speaking, or listen. CSM-1B speaks its words.

Research-only weights. See `WEIGHTS_LICENSE.md`.

## Setup

```bash
uv sync
```

## Download weights

```bash
uv run parley download walln/parley --out checkpoints/parley
```

## Serve

```bash
uv run parley serve --checkpoint checkpoints/parley --prompt "You are a helpful assistant."
```

## License

Code: Apache-2.0. Weights: research-only (CC-BY-NC 4.0).