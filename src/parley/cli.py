"""Parley command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer

from parley.serve import run_server

app = typer.Typer(help="Parley: a full-duplex speech model.")


@app.command()
def serve(
    checkpoint: Path = typer.Option(..., help="Path to a Parley checkpoint directory."),
    prompt: str = typer.Option("You are a helpful assistant.", help="System prompt."),
    voice_wav: Path = typer.Option(Path("assets/voice.wav"), help="Reference voice WAV (24 kHz mono)."),
    voice_text: str = typer.Option("Hello there, how are you today?", help="Transcript of the reference voice."),
    port: int = typer.Option(8766, help="Port to serve on."),
    session_seconds: float = typer.Option(30.0, help="Maximum session length in seconds."),
    temperature: float = typer.Option(0.8, help="Text sampling temperature."),
    stop_threshold: float = typer.Option(0.5, help="STOP probability threshold."),
    gate_hold_frames: int = typer.Option(12, help="Noise-gate hold in frames."),
):
    run_server(
        checkpoint=checkpoint,
        prompt=prompt,
        voice_wav=voice_wav,
        voice_text=voice_text,
        port=port,
        session_seconds=session_seconds,
        temperature=temperature,
        stop_threshold=stop_threshold,
        gate_hold_frames=gate_hold_frames,
    )


@app.command()
def download(repo: str, out: Path = Path("checkpoints/parley")):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo, local_dir=str(out))
    typer.echo(f"Downloaded {repo} to {out}")


def main():
    app()


if __name__ == "__main__":
    main()
