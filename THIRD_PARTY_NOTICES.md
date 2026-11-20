# Third-party notices

## Code

| Project | Revision | License | Used for |
| --- | --- | --- | --- |
| [MiniMind](https://github.com/jingyaogong/minimind/tree/f659b55761b754d306bd140573493a6543cafd7f) (jingyaogong and contributors) | `f659b55761b754d306bd140573493a6543cafd7f` | Apache-2.0, [text](LICENSE) | The dense decoder in [`src/finllm/model.py`](src/finllm/model.py) is adapted from [`model/model_minimind.py`](https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/model/model_minimind.py) |

RMSNorm, rotary positions, grouped-query attention, SwiGLU and tied input and output embeddings follow the pinned
MiniMind architecture. The changes made here in 2026: a native PyTorch dataclass configuration; a dense default of
24,255,744 parameters; no Transformers dependency, mixture of experts, generation cache or internal shift of the
next-token loss; explicit causal and padding masks; input checks; and tests for causal isolation, padding
behavior and finite gradients. No pretrained MiniMind weights or tokenizer are included or downloaded. The
attribution does not imply endorsement by the upstream authors.

PyTorch, NumPy, Hugging Face Tokenizers and scikit-learn are dependencies under their own licenses and are not
vendored.

## Text

The repository holds code and aggregate measurements. It contains no corpus text, no reviewed headlines and no
model weights. The Trial was trained on text from three sources, each read at a fixed revision.

| Source | Revision | Used for |
| --- | --- | --- |
| [English Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md), Wikimedia's dataset | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | General text: a prefix of the English split |
| [Minutes of the Federal Open Market Committee](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm), Board of Governors of the Federal Reserve System | Meetings from 2011 to 2025 | Financial text |
| [SEntFiN 1.0](https://github.com/pyRis/SEntFiN/tree/eba43610fe3cb57a3e5773cd6c037da1f8992ad5), Sinha, Kedas, Kumar and Malo | `eba43610fe3cb57a3e5773cd6c037da1f8992ad5` | The pool of headlines with named targets: all 10,686 distinct headlines were excluded from the tokenizer and the language corpora, and 2,000 were reviewed for investment stance |

The rows of [`data/examples/annotation_template.jsonl`](data/examples/annotation_template.jsonl) were written for
this repository to show the record format.
