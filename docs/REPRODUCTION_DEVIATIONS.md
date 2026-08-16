# Eraser4RAG v2 Reproduction Deviations

This document separates behavior observed in the public author snapshot from
paper-v2 alignment, compatibility reconstruction, and engineering-only work.
The source baseline is commit `2fb526451ad49735cd6d1826b8ff882e40c49a5f`.

## Labels

- `[AUTHOR-CODE]`: behavior directly present in the public snapshot.
- `[PAPER-V2]`: a setting explicitly stated by paper v2 but absent or stale in
  the public snapshot.
- `[RECONSTRUCTION]`: behavior rebuilt to make the public snapshot executable,
  compatible, or internally consistent. It is not claimed as an unpublished
  author setting.
- `[ENGINEERING]`: paths, CLI, validation, logging, safety, and tests that do
  not intentionally change the algorithmic objective.

## Preserved author behavior

| Area | Classification | Treatment |
|---|---|---|
| Rewriter prompt and eight public/private special tokens | `[AUTHOR-CODE]` | Preserved and centralized in shared serializers. |
| PPO fixed reference | `[AUTHOR-CODE]` | Remains the original document-local public/private triples. GPT-rewritten extractions are never used as the PPO fixed reference. |
| PPO dynamic extraction | `[AUTHOR-CODE]` + `[RECONSTRUCTION]` | Only the current policy rewrite is dynamically passed through ReLiK. Its triples now use the same unordered entity-pair first-relation rule as the fixed-reference extraction and mandatory consistency gate; the released PPO helper itself removed only exact duplicates/self-loops. |
| Reward | `[AUTHOR-CODE]` + `[PAPER-V2]` | Kept exactly as `r_pub * exp(-p * r_pri)`. |
| ReLiK `use_nme=True` | `[AUTHOR-CODE]` | Default preserved, with a reliable CLI switch for diagnosis. |
| Relation-extraction duplicate rule | `[AUTHOR-CODE]` | The first relation for an unordered subject/object pair is retained; later relations for that pair are dropped. The consistency gate applies the same rule. |
| Graph connectivity | `[AUTHOR-CODE]` | Undirected connectivity and the author's substring short-form heuristic are retained and documented; they are not equivalent to semantic inference. |
| RL context filter | `[AUTHOR-CODE]` | Defaults preserve `local public >= 5` and `local private >= 2`, now exposed and logged. |

## Paper-v2 alignment

| Change | Classification | Reason |
|---|---|---|
| Add HotpotQA as the fourth PPO data interface | `[PAPER-V2]` | Paper v2 states RL uses PopQA, TriviaQA, NQ, and HotpotQA. |
| Set PPO discount `gamma=0.99` | `[PAPER-V2]` | The public code only set the LR scheduler gamma; TRL PPO gamma otherwise remained 1.0. |
| Cap privacy penalty at 40 | `[PAPER-V2]` | The public `<= 40` increment condition reached 45 at step 1750. |
| Boundary convention | `[RECONSTRUCTION]` | Outer steps 1–349 use 20; 350–699 use 25; 700–1049 use 30; 1050–1399 use 35; step 1400 onward uses 40. This convention is explicit in tests and logs. |
| Formal SFT defaults | `[PAPER-V2]` | Flan-T5-large, 3 epochs, lr `5e-5`, input 1300, target 128. |

## Compatibility reconstruction

### Environments

- `[RECONSTRUCTION]` The original single requirements file is archived but is
  not treated as installable: Coreferee conflicts with ReLiK's spaCy range,
  and the pinned Transformers/scikit-learn versions conflict with ReLiK 1.0.7.
- `[RECONSTRUCTION]` Two Python 3.10.14 environments are defined:
  `eraser-coref` (Coreferee/spaCy 3.5) and `eraser-main`
  (ReLiK/Transformers 4.41.2/TRL 0.11.4).
- `[RECONSTRUCTION]` Torch 2.3.1 is installed separately for the actual AutoDL
  CUDA image. vLLM and Coreferee are not installed into `eraser-main`.

### Data and triple handling

- `[RECONSTRUCTION]` The missing `utilities.process_private_trps` dependency is
  replaced by strict shared neutral/public/private serialization helpers.
- `[RECONSTRUCTION]` The SFT dataset has no query/group identifier. Consecutive
  rows with identical, order-sensitive global public/private graphs are grouped
  only for PPO engineering smoke data. This reconstruction is marked
  `source=regrouped_author_sft` and is not formal PPO data.
- `[RECONSTRUCTION]` JSON object document IDs are normalized to strings at
  alignment boundaries to avoid integer/string mismatches after serialization.
- `[RECONSTRUCTION]` Deterministic graph sampling uses one seeded RNG rather
  than resetting seed 42 inside every query.
- `[RECONSTRUCTION]` A sampled triple rejected because it would harm QA returns
  to the public candidate set instead of disappearing from both partitions.
- `[RECONSTRUCTION]` When a public triple is moved into privacy, the private
  graph is rebuilt so later connectivity checks see the expanded graph. This
  implements the stated bidirectional filtering intent; it is not claimed as
  byte-identical hidden author code.
- `[RECONSTRUCTION]` An empty public reference is assigned `r_pub=1`; an empty
  private reference is assigned `r_pri=0`. Formal author PPO defaults filter out
  these cases, but explicit definitions prevent division-by-zero in validators
  and smoke tests.
- `[RECONSTRUCTION]` The special-set rewrite output now retains its source
  `ctxs`, and inference-attack output uses the global `privacy` key. These
  repair released producer/consumer schema mismatches without changing model
  generation.

### PPO execution

- `[RECONSTRUCTION]` `max_steps` is defined as exact outer rollout/update
  batches. The loop stops at that count rather than relying on TRL's derived
  `total_ppo_epochs` and accidentally overshooting.
- `[RECONSTRUCTION]` Formal PPO defaults to 200,000 outer updates because the
  paper/public snapshot does not report a definitive stopping condition. This
  value is explicit in configuration and must be reported as an experiment
  choice, not an author hyperparameter.
- `[RECONSTRUCTION]` Gradient accumulation is one so batch 16 / mini-batch 8
  has an unambiguous effective meaning under TRL 0.11.4.
- `[RECONSTRUCTION]` Formal RL input percentage defaults to 1.0. The author
  snapshot's 0.6 default appeared to be an unreported debugging filter and can
  still be requested explicitly.
- `[RECONSTRUCTION]` Rewriter tokenizer and policy checkpoints are saved
  together after registering all eight tokens. The base tokenizer directory is
  never overwritten.
- `[RECONSTRUCTION]` Non-dry PPO execution now requires a valid ReLiK
  consistency report and sample manifest. A warning report requires an
  explicit CLI acknowledgement so the measurement deviation cannot be missed.

## Engineering changes

- `[ENGINEERING]` Added deterministic seeds, UTF-8 I/O, parent-directory
  creation, safe JSONL extensions, partial-batch handling, explicit errors,
  output joins with `pathlib`, and reliable boolean flags.
- `[ENGINEERING]` Added `--max-samples`, SFT smoke/resume, PPO dry-run,
  generate-only, reward-only, one/ten-step controls, and model/device/path
  parameters.
- `[ENGINEERING]` Text samples that may still contain private knowledge are not
  logged unless explicitly requested.
- `[ENGINEERING]` Added SFT/RL schema validators, fixed sample manifests,
  environment/model checks, model download stages, pytest isolation, and an
  AutoDL runbook.
- `[ENGINEERING]` Model downloads require an explicit revision (or an explicit
  one-time floating-head acknowledgement), resolve that revision to an
  immutable Hub commit, and write a download manifest.
- `[ENGINEERING]` The ReLiK value `0.80` is only an engineering warning
  threshold. It is never described as a paper threshold or formal privacy
  guarantee.

## Completed reduced AutoDL experiment

The repository now includes one completed PopQA + HotpotQA core experiment,
recorded in `results/two_dataset_b8_3000/`. This run is deliberately narrower
than the paper's four-dataset setup and does not include downstream Llama-3 RAG
QA accuracy.

- Formal SFT used the 23,074-row author PopQA input, Flan-T5-large, three
  epochs, learning rate `5e-5`, input length 1300, and target length 128.
- The fixed 50- and 500-sample ReLiK gates passed on AutoDL. The 500-sample
  public/private micro recalls were `0.9536/0.9653`.
- Reduced PPO used 5,000 PopQA plus 5,000 HotpotQA training records, batch 8,
  mini-batch 4, four PPO epochs, `gamma=0.99`, and 3,000 outer updates. The
  privacy penalty followed `20,25,30,35,40` and remained 40 from step 1400.
- Training and all six final-policy ReLiK evaluations completed on an RTX 4090
  48 GiB instance. The final weight SHA-256 is
  `489453a9d640e4914462fa7bd2e221bff274b956fc0390e7413ee54f8920569f`.
- The 3,000-update stopping point is this reproduction's declared compute
  choice. It is not presented as an author-reported paper hyperparameter.

## Known unresolved limitations

1. GPU model work cannot be reproduced in the local model-free preparation
   environment. The ReLiK gates, SFT, reduced PPO, and final metrics were run
   on AutoDL; large artifacts remain on its persistent data disk and are
   represented publicly by settings, counts, results, and hashes.
2. TriviaQA and NQ-Open were not included in the completed reduced experiment.
   Downstream Llama-3 RAG QA accuracy also remains unimplemented, so this is
   not a complete reproduction of the paper's tables.
3. The public repository does not contain a complete, pinned Wikipedia
   retrieval/index construction entry point. The formal four-dataset corpus
   cannot be reconstructed from QA downloads alone.
4. The SFT file lacks query IDs and document IDs; reconstructed groups are an
   engineering inference from consecutive repeated graphs.
5. Coreferee's author-style token-by-token reconstruction can alter spacing and
   punctuation. The script records provenance but does not claim lossless text
   reconstruction.
6. The substring entity heuristic can over-match short strings. It is retained
   for author compatibility and should be separately evaluated on noisy data.
7. The same ReLiK family participates in reference construction, PPO reward,
   and evaluation; consistency validation measures this circularity but cannot
   remove it.
8. Graph connectivity is a structural proxy, not a formal privacy guarantee.
9. Model commit revisions were not stated by the paper/public snapshot. This
   reconstruction freezes current Hub commits in
   `configs/reproduction.yaml` as of 2026-08-12; these are engineering
   reproducibility choices, not claims about the authors' original snapshots.
