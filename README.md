# Eraser4RAG

Source code for *Learning to Erase Private Knowledge from Multi-Documents for
Retrieval-Augmented Large Language Models*.

This branch contains a paper-v2 reproduction preparation layer around the
released research snapshot. Start with:

- [AutoDL runbook](docs/AUTODL_RUNBOOK.md) for the staged GPU workflow;
- [reproduction deviations](docs/REPRODUCTION_DEVIATIONS.md) for the exact
  distinction between author behavior, paper-v2 alignment, reconstruction,
  and engineering changes;

Local execution notes and machine-specific status are kept under the ignored
`plan/` directory and are intentionally not published.

## Environment

Do **not** install the root `requirements.txt` as one environment. It is kept as
the original author snapshot and contains incompatible Coreferee/ReLiK spaCy
constraints. The reconstruction uses two Python 3.10.14 environments:

- `requirements/eraser-main.txt` for SFT, ReLiK, PPO, and evaluation;
- `requirements/eraser-coref.txt` only for Coreferee preprocessing.

Torch 2.3.1 must be installed separately with the CUDA build matching the
selected AutoDL image. See the runbook before installing models or starting
training.

## Local model-free checks

```bash
python -m pytest -q
python scripts/validate_sft_dataset.py \
  --data dataset/constructed_dataset/popqa_10_25_filtered_new.jsonl \
  --output-dir outputs/validation
```

`finetune_rewrite_doc.py` performs Flan-T5 SFT. `RL_train.py` performs PPO and
enforces the ReLiK consistency report before any non-dry run. The scripts under
`dataset/`, `utils/`, and `scripts/` cover preprocessing, schema validation,
fixed consistency sampling, smoke-corpus reconstruction, model acquisition,
and environment checks.

The public snapshot still lacks a complete pinned Wikipedia
retrieval/index-build entry point and downstream Llama-3 QA evaluation. Those
gaps must be resolved and recorded before claiming full paper-table
reproduction.

## References

The released retrieval code refers to
[Self-RAG](https://github.com/AkariAsai/self-rag), and PPO uses
[Hugging Face TRL](https://github.com/huggingface/trl).
