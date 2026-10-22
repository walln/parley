# Architecture

Parley is Gemma 4 12B with LoRA adapters, plus CSM-1B as a separate talker.
Gemma hears 16 kHz audio (40 ms slices, no encoder) and writes text/control tokens
on an 80 ms clock: `SIL` (listening), `EPS` (still speaking), or words. CSM-1B
speaks those words in chunks through a playout queue. There is no depth head and no
audio feedback: Gemma never predicts audio.

See the blog post for the full story.