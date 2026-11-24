# Plan: Authorship Style Transfer with Proof ("rewrite D's document as A / B / C")

## Context
The goal: learn the writing styles of persons A, B and C. A new person D then submits a document, and the system rewrites the **whole document** in the chosen person's style. The meaning must stay the same, and the output must come with **proof** (style-match scores) that it really reads like that person.

The current CATbot project (`model.py`, `generate.py`, `app.py`) is a word-level **Markov chain**. It can only *generate* random text that sounds like one speaker. It cannot take an input document and keep its meaning, so it can't do this task. We keep its useful parts (the cleaning in `clean.py`, the copy check `copy_stats()` in `generate.py`, and the Streamlit UI pattern in `app.py`) and build the new system next to it.

Decisions from the user: use a **public dataset**; the proof is **style-match scores** from a similarity function; run on the local machine for now and **decide on training later**.

**Hardware finding:** this machine has an AMD Radeon Vega *integrated* GPU (Cezanne), 14 GB RAM, no CUDA, and PyTorch isn't installed. It can *run* a quantized 3B–7B model through Ollama or llama.cpp (Vulkan or CPU), but it **cannot fine-tune**. Fine-tuning (Phase 2) needs Colab, Kaggle or another NVIDIA machine. That's why the design is model-agnostic.

## Recommended architecture: Neutralize → Restyle → Verify

```
D's document
  │ 1. split into paragraphs
  ▼
[Neutralizer]  LLM paraphrases each paragraph into plain, style-free text (content only)
  ▼
[Retriever]    finds the k most similar real passages from the chosen author (A/B/C) by topic
  ▼
[Restyler]     Phase 1: LLM + few-shot prompt (author style card + retrieved passages)
               Phase 2: the same LLM + a per-author LoRA adapter (trained neutral→author)
  │  generate N candidates per paragraph
  ▼
[Verifier]     scores each candidate: style similarity to target, meaning kept, not copied
  │  keep the best candidate; retry if it falls below threshold
  ▼
Full rewritten document + Proof Report (per paragraph and whole document)
```

Why this design:
- Neutralizing first removes D's own style, so the output doesn't end up as a mix of D's style and A's style. This is the core idea of STRAP (Krishna et al. 2020).
- Few-shot prompting with an LLM plus retrieval is the strongest no-training baseline (STYLL, Patel et al.). Fine-tuning with LoRA later adds real "learning" of the person.
- Training data for LoRA comes free: neutralize each real passage from A, then train the pair (neutral → A's original). This gives pseudo-parallel data, the same trick STRAP and ASTRAPOP use.
- The verifier is both the quality filter and the **proof generator**.

## Models
| Role | Model | Notes |
|---|---|---|
| Neutralizer and Restyler LLM | **Qwen2.5-3B/7B-Instruct** (or Llama-3.2-3B-Instruct), GGUF Q4 through **Ollama** | Runs locally on CPU or Vulkan. 7B Q4 needs about 5 GB RAM. Accessed via an OpenAI-compatible client, so the backend can be swapped. |
| Retrieval embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Matches passages by topic or content. |
| **Style similarity (main proof)** | **`StyleDistance/styledistance`** | Content-independent style embedding (NAACL 2025). The model card lists style-transfer evaluation as a use. |
| Second style proof | **LUAR** (`rrivera1849/LUAR-MUD`) | Authorship embedding (EMNLP 2021). Gives author-level similarity. |
| Authorship classifier | Logistic regression on stylometric features + StyleDistance vectors, classes {A, B, C, other} | Gives a "P(author = A) = 0.87" score. |
| Meaning preservation | `all-mpnet-base-v2` cosine (+ BERTScore optional) | Needed, or the model can "cheat" by pasting A's text. |
| Phase 2 fine-tuning | QLoRA (Unsloth / PEFT), r=16, one adapter per author, on Colab T4 or Kaggle | Decided later; the pipeline doesn't change. |

## Dataset
**Primary: Blog Authorship Corpus** (HF `barilan/blog_authorship_corpus`: 19k bloggers, 681k posts, 140M words; non-commercial research use).
- Pick the 3 most prolific bloggers with **clearly different** styles (check by the distance between their StyleDistance centroids) as A, B and C. Use about 50k+ words each.
- Person **D** is any other blogger. Their posts are the input documents and test set.
- Split per author: 80% for the style profile and LoRA training, 20% held out (used for the classifier and to sanity-check the scores).
- Optional harder benchmark: **Reuters C50** (50 journalists, 100 articles each). The styles are similar there, so it's a tougher test.

## Proof report (what the user receives)
For each paragraph and for the whole document:
1. **Style similarity**: cosine(StyleDistance(output), centroid_A), shown next to the same score for D's original text, plus the gain (Δ).
2. **Author-embedding similarity** (LUAR) against A, B and C. The output should be closest to the chosen author.
3. **Classifier verdict**: P(A), P(B), P(C), P(other).
4. **Meaning preserved**: semantic similarity between D's original and the output (target ≥ 0.80).
5. **Not copied**: % of 8-gram overlap with A's corpus, reusing `copy_stats()` from [generate.py](generate.py).
6. A small stylometric table (average sentence length, type-token ratio, punctuation rate, top function words) for D, A and the output.
Exported as Markdown/HTML along with the rewritten document.

**Calibration:** report how held-out real text by A scores. That gives the "real A" baseline, so a score like 0.78 has a meaning.

## Project layout (new folder `styleshift/` inside CATbot_Complete)
- `styleshift/data_prep.py`: downloads the corpus, picks the authors, cleans text (reuse `clean_text()` / `read_text()` from [clean.py](clean.py)), splits into paragraphs and train/test sets.
- `styleshift/profile.py`: builds a profile per author: StyleDistance centroid, LUAR embedding, stylometric stats, passage index for retrieval, and an LLM-written "style card".
- `styleshift/llm.py`: OpenAI-compatible client (Ollama now; a LoRA-served or API model later).
- `styleshift/transfer.py`: neutralize → retrieve → restyle (N candidates) → pick the best.
- `styleshift/verify.py`: all the proof metrics and the classifier; produces the proof report.
- `styleshift/make_pairs.py`: (Phase 2) builds the neutral→author JSONL used for LoRA.
- `styleshift/train_lora.ipynb`: (Phase 2) Colab/Kaggle QLoRA notebook.
- `styleshift/app.py`: Streamlit UI that follows [app.py](app.py): upload D's document, choose A/B/C, view the output and the proof report, then download.
- `styleshift/requirements.txt`: sentence-transformers, transformers, torch (CPU), scikit-learn, datasets, openai, streamlit.

## Phases
1. **Data and profiles**: data_prep + profile. Check that A, B and C are distinguishable (the classifier gets >85% accuracy on held-out data).
2. **Verifier first**: build verify.py and calibrate it on real held-out text, so we can measure every later step.
3. **Prompting baseline**: transfer.py with Ollama Qwen2.5-7B. Measure style Δ and meaning scores on about 30 D documents.
4. **Phase 2 (decided later)**: make pseudo-parallel pairs, train LoRA adapters on Colab or Kaggle, then compare against the baseline using the same verifier.
5. **UI and report**: Streamlit app plus an export of the proof report.

## Existing work to reuse or cite
- STRAP, style transfer as paraphrase: github.com/martiansideofthemoon/style-transfer-paraphrase
- ASTRAPOP, authorship style transfer with PPO/DPO/CPO: github.com/isi-nlp/ASTRAPOP
- TinyStyler, an 800M model with authorship embeddings that beats GPT-4: huggingface.co/tinystyler/tinystyler (arXiv 2406.15586). A good extra baseline.
- STYLL, few-shot low-resource authorship transfer: arXiv 2212.08986
- LUAR: github.com/llnl/luar · StyleDistance: huggingface.co/StyleDistance/styledistance
- Style LoRA write-ups: Unsloth + Llama-3.2-3B on your own writing (dev.to)

## Verification
- `python styleshift/data_prep.py`: prints the authors chosen, word counts and the split.
- `python styleshift/verify.py --calibrate`: held-out real text by A should score highest against A. Classifier accuracy is printed.
- `python styleshift/transfer.py --input sample_D.txt --target A`: prints the rewritten document and the proof report. Expected: style similarity to A rises clearly above D's original score, meaning similarity ≥ 0.80, copy overlap < 20%, and the classifier picks A.
- Batch evaluation on about 30 held-out D documents × 3 targets: a table of average style Δ, meaning score and transfer accuracy.
- `streamlit run styleshift/app.py`: end-to-end check in the UI.
