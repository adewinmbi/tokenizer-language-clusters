#!/usr/bin/env python3
"""
Experiment: Do LLaMA 3 8B tokenizer embeddings cluster morphosyntactic concepts
cross-lingually?

For each grammatical concept (e.g. Number=Sing, Gender=Masc) in each language,
we:
  1. Parse UD CoNLL-U treebanks to get (word, features) pairs.
  2. Tokenize each word with the LLaMA 3 8B tokenizer.
  3. Assign each subword token its dominant feature value via max-pooling
     (most-common label across all words that token appears in).
  4. Load the token embedding matrix from the model.
  5. Compute the centroid and inertia for each (concept, language) group.
  6. Compare combined inertia and centroid cosine similarity for:
       - same concept, different languages  (hypothesis: tight / high sim)
       - different concepts, different languages  (hypothesis: loose / lower sim)
       - different concepts, same language  (additional baseline)

Usage:
  python experiment.py --smoke                # lightweight end-to-end check
  python experiment.py                        # full experiment, all 23 languages
  python experiment.py --output results.json  # save results to JSON
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

# ---------------------------------------------------------------------------
# Language → treebank mapping (one canonical treebank per language)
# ---------------------------------------------------------------------------
TREEBANKS: dict[str, tuple[str, str]] = {
    "Arabic":     ("UD_Arabic-PADT",       "ar_padt"),
    "Chinese":    ("UD_Chinese-GSD",        "zh_gsd"),
    "Czech":      ("UD_Czech-PDT",          "cs_pdt"),
    "Dutch":      ("UD_Dutch-Alpino",       "nl_alpino"),
    "English":    ("UD_English-EWT",        "en_ewt"),
    "French":     ("UD_French-GSD",         "fr_gsd"),
    "German":     ("UD_German-GSD",         "de_gsd"),
    "Greek":      ("UD_Greek-GDT",          "el_gdt"),
    "Hebrew":     ("UD_Hebrew-HTB",         "he_htb"),
    "Hindi":      ("UD_Hindi-HDTB",         "hi_hdtb"),
    "Indonesian": ("UD_Indonesian-GSD",     "id_gsd"),
    "Italian":    ("UD_Italian-ISDT",       "it_isdt"),
    "Japanese":   ("UD_Japanese-GSD",       "ja_gsd"),
    "Korean":     ("UD_Korean-GSD",         "ko_gsd"),
    "Persian":    ("UD_Persian-Seraji",     "fa_seraji"),
    "Polish":     ("UD_Polish-PDB",         "pl_pdb"),
    "Portuguese": ("UD_Portuguese-Bosque",  "pt_bosque"),
    "Romanian":   ("UD_Romanian-RRT",       "ro_rrt"),
    "Russian":    ("UD_Russian-SynTagRus",  "ru_syntagrus"),
    "Spanish":    ("UD_Spanish-AnCora",     "es_ancora"),
    "Turkish":    ("UD_Turkish-IMST",       "tr_imst"),
    "Ukrainian":  ("UD_Ukrainian-IU",       "uk_iu"),
    "Vietnamese": ("UD_Vietnamese-VTB",     "vi_vtb"),
}

SMOKE_LANGUAGES = ["English", "French", "German"]
SMOKE_SPLITS = ["test"]
FULL_SPLITS = ["train", "dev", "test"]

DEFAULT_MIN_TOKENS = 5       # minimum distinct tokens per (concept, language) group
DEFAULT_MIN_CONFIDENCE = 0.5 # dominant feature value must be this fraction of occurrences


# ---------------------------------------------------------------------------
# CoNLL-U parsing
# ---------------------------------------------------------------------------

def parse_conllu(path: Path, max_sentences: Optional[int] = None):
    """Yield (word_form, feature_dict) for every annotated token in a CoNLL-U file."""
    sentence_count = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                sentence_count += 1
                if max_sentences is not None and sentence_count >= max_sentences:
                    return
                continue
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 10:
                continue
            tok_id = fields[0]
            if "-" in tok_id or "." in tok_id:   # skip multi-word / empty nodes
                continue
            feats_str = fields[5]
            if feats_str == "_":
                continue
            feats: dict[str, str] = {}
            for feat in feats_str.split("|"):
                if "=" in feat:
                    k, v = feat.split("=", 1)
                    feats[k] = v
            if feats:
                yield fields[1], feats   # (word_form, feature_dict)


def load_language(lang: str, splits: list[str], data_dir: Path,
                  max_sentences: Optional[int] = None) -> list[tuple[str, dict[str, str]]]:
    treebank_dir, prefix = TREEBANKS[lang]
    pairs: list[tuple[str, dict[str, str]]] = []
    for split in splits:
        path = data_dir / treebank_dir / f"{prefix}-ud-{split}.conllu"
        if path.exists():
            pairs.extend(parse_conllu(path, max_sentences))
    return pairs


# ---------------------------------------------------------------------------
# Token → feature mapping (with max-pooling)
# ---------------------------------------------------------------------------

def build_token_feature_map(
    tokenizer,
    lang_words: dict[str, list[tuple[str, dict[str, str]]]],
    min_tokens: int = DEFAULT_MIN_TOKENS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> dict[str, dict[str, dict[str, set[int]]]]:
    """
    Returns {lang -> {feat_cat -> {feat_val -> set_of_token_ids}}}.

    For each subword token, we max-pool: the dominant feature value (by raw
    occurrence count across all words containing that token) is kept, provided
    it exceeds min_confidence of total observations for that feature category.
    Groups with fewer than min_tokens distinct token IDs are discarded.
    """
    # raw[lang][token_id][feat_cat][feat_val] = count
    raw: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))

    for lang, word_feats in lang_words.items():
        for word, feats in word_feats:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            for tid in token_ids:
                for feat_cat, feat_val in feats.items():
                    raw[lang][tid][feat_cat][feat_val] += 1

    # pooled[lang][feat_cat][feat_val] = set of token_ids
    pooled: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    for lang, token_data in raw.items():
        for tid, feat_cats in token_data.items():
            for feat_cat, val_counts in feat_cats.items():
                total = sum(val_counts.values())
                dominant_val, dominant_cnt = val_counts.most_common(1)[0]
                if dominant_cnt / total >= min_confidence:
                    pooled[lang][feat_cat][dominant_val].add(tid)

    # Filter to groups with enough tokens
    result: dict = {}
    for lang, feat_cats in pooled.items():
        result[lang] = {}
        for feat_cat, feat_vals in feat_cats.items():
            filtered = {fv: tids for fv, tids in feat_vals.items()
                        if len(tids) >= min_tokens}
            if filtered:
                result[lang][feat_cat] = filtered

    return result


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def load_embeddings(model_name: str) -> np.ndarray:
    """
    Efficiently download and load only the token embedding matrix from the
    LLaMA 3 8B model.  Tries the sharded safetensors index first; falls back
    to a single-file safetensors, then to AutoModel.
    """
    import json as _json
    from huggingface_hub import hf_hub_download

    def _from_safetensors(path: str) -> np.ndarray:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            return f.get_tensor("model.embed_tokens.weight").float().numpy()

    # Try sharded index
    try:
        print(f"  Fetching shard index for {model_name} …")
        index_path = hf_hub_download(model_name, "model.safetensors.index.json")
        with open(index_path) as f:
            index = _json.load(f)
        shard_name = index["weight_map"]["model.embed_tokens.weight"]
        print(f"  Downloading {shard_name} …")
        shard_path = hf_hub_download(model_name, shard_name)
        embeddings = _from_safetensors(shard_path)
        print(f"  Loaded embeddings: shape {embeddings.shape}")
        return embeddings
    except Exception as e:
        print(f"  Sharded load failed ({e}); trying single file …")

    # Try single safetensors file
    try:
        shard_path = hf_hub_download(model_name, "model.safetensors")
        embeddings = _from_safetensors(shard_path)
        print(f"  Loaded embeddings: shape {embeddings.shape}")
        return embeddings
    except Exception as e:
        print(f"  Single-file load failed ({e}); falling back to AutoModel …")

    # Last resort: load full model (slow but reliable)
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32,
                                      device_map="cpu")
    embeddings = model.embed_tokens.weight.detach().float().numpy()
    del model
    print(f"  Loaded embeddings via AutoModel: shape {embeddings.shape}")
    return embeddings


# ---------------------------------------------------------------------------
# Cluster geometry helpers
# ---------------------------------------------------------------------------

def centroid_and_inertia(
    token_ids: set[int], embeddings: np.ndarray
) -> tuple[Optional[np.ndarray], Optional[float]]:
    valid = [tid for tid in token_ids if tid < len(embeddings)]
    if not valid:
        return None, None
    vecs = embeddings[valid]          # (n, d)
    centroid = vecs.mean(axis=0)      # (d,)
    inertia = float(np.sum((vecs - centroid) ** 2))
    return centroid, inertia


# ---------------------------------------------------------------------------
# Cross-lingual analysis
# ---------------------------------------------------------------------------

def analyze(
    token_map: dict,
    embeddings: np.ndarray,
) -> list[dict]:
    """
    Returns a flat list of comparison records, each with keys:
      type, feat_cat, feat_val_A, feat_val_B, lang_A, lang_B,
      combined_inertia, centroid_cosine_sim, n_tokens_A, n_tokens_B
    """
    # Pre-compute per-group stats
    # stats[lang][feat_cat][feat_val] = {centroid, inertia, n}
    stats: dict = {}
    for lang, feat_cats in token_map.items():
        stats[lang] = {}
        for feat_cat, feat_vals in feat_cats.items():
            stats[lang][feat_cat] = {}
            for feat_val, tids in feat_vals.items():
                c, inertia = centroid_and_inertia(tids, embeddings)
                if c is not None:
                    stats[lang][feat_cat][feat_val] = {
                        "centroid": c, "inertia": inertia, "n": len(tids)
                    }

    comparisons: list[dict] = []

    # Collect all feature categories that appear in at least 2 languages
    all_feat_cats: set[str] = set()
    for feat_cats in token_map.values():
        all_feat_cats.update(feat_cats)

    for feat_cat in sorted(all_feat_cats):
        # All (lang, feat_val) groups for this feature category
        groups: list[tuple[str, str]] = []   # (lang, feat_val)
        for lang in token_map:
            if feat_cat in stats.get(lang, {}):
                for feat_val in stats[lang][feat_cat]:
                    groups.append((lang, feat_val))

        # Pairwise comparisons between all groups
        for i, (lang_A, val_A) in enumerate(groups):
            for lang_B, val_B in groups[i + 1:]:
                data_A = stats[lang_A][feat_cat][val_A]
                data_B = stats[lang_B][feat_cat][val_B]

                # Combined inertia of merging the two groups
                ids_A = token_map[lang_A][feat_cat][val_A]
                ids_B = token_map[lang_B][feat_cat][val_B]
                _, combined_inertia = centroid_and_inertia(ids_A | ids_B, embeddings)

                cos_sim = float(sk_cosine_similarity(
                    data_A["centroid"].reshape(1, -1),
                    data_B["centroid"].reshape(1, -1),
                )[0, 0])

                same_lang = (lang_A == lang_B)
                same_val  = (val_A  == val_B)

                if same_val and not same_lang:
                    ctype = "same_concept_cross_lang"
                elif not same_val and not same_lang:
                    ctype = "diff_concept_cross_lang"
                elif not same_val and same_lang:
                    ctype = "diff_concept_same_lang"
                else:
                    continue   # same concept, same language – not informative

                comparisons.append({
                    "type":               ctype,
                    "feat_cat":           feat_cat,
                    "feat_val_A":         val_A,
                    "feat_val_B":         val_B,
                    "lang_A":             lang_A,
                    "lang_B":             lang_B,
                    "combined_inertia":   combined_inertia,
                    "centroid_cosine_sim": cos_sim,
                    "n_tokens_A":         data_A["n"],
                    "n_tokens_B":         data_B["n"],
                })

    return comparisons


# ---------------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------------

def _agg(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    return f"mean={statistics.mean(vals):.4f}  median={statistics.median(vals):.4f}  n={len(vals)}"


def print_results(comparisons: list[dict]) -> None:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for c in comparisons:
        by_type[c["type"]].append(c)

    feat_cats = sorted({c["feat_cat"] for c in comparisons})

    print("\n" + "=" * 72)
    print("RESULTS BY FEATURE CATEGORY")
    print("=" * 72)

    TYPE_LABELS = {
        "same_concept_cross_lang": "Same concept / cross-language",
        "diff_concept_cross_lang": "Diff concept / cross-language",
        "diff_concept_same_lang":  "Diff concept / same-language ",
    }

    for feat_cat in feat_cats:
        print(f"\n  [{feat_cat}]")
        for ctype, label in TYPE_LABELS.items():
            rows = [c for c in by_type[ctype] if c["feat_cat"] == feat_cat]
            if not rows:
                continue
            cos_sims  = [r["centroid_cosine_sim"] for r in rows]
            inertias  = [r["combined_inertia"] for r in rows
                         if r["combined_inertia"] is not None]
            print(f"    {label}")
            print(f"      cosine sim:       {_agg(cos_sims)}")
            print(f"      combined inertia: {_agg(inertias)}")

    print("\n" + "=" * 72)
    print("HYPOTHESIS TESTS  (higher cosine sim = tighter cluster)")
    print("=" * 72)

    sc  = [c["centroid_cosine_sim"] for c in by_type["same_concept_cross_lang"]]
    dcl = [c["centroid_cosine_sim"] for c in by_type["diff_concept_cross_lang"]]
    dsl = [c["centroid_cosine_sim"] for c in by_type["diff_concept_same_lang"]]

    def _verdict(a: list, b: list, a_label: str, b_label: str) -> None:
        if not a or not b:
            print("  insufficient data")
            return
        ma, mb = statistics.mean(a), statistics.mean(b)
        verdict = "SUPPORTED" if ma > mb else "NOT SUPPORTED"
        print(f"  {a_label}: {ma:.4f}")
        print(f"  {b_label}: {mb:.4f}")
        print(f"  → {verdict}")

    print("\nH1: Same-concept cross-language clusters are tighter than")
    print("    different-concept cross-language clusters.")
    _verdict(sc, dcl,
             "same concept / cross-lang  avg cos sim",
             "diff concept / cross-lang  avg cos sim")

    print("\nH2: Same-concept cross-language clusters are tighter than")
    print("    different-concept within-language clusters.")
    _verdict(sc, dsl,
             "same concept / cross-lang  avg cos sim",
             "diff concept / same-lang   avg cos sim")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenizer cross-lingual morphosyntactic clustering experiment"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Lightweight smoke test: English/French/German, test split, 200 sentences",
    )
    parser.add_argument(
        "--data-dir", default="data/universal_dependencies",
        help="Path to Universal Dependencies root directory",
    )
    parser.add_argument(
        "--model", default="meta-llama/Meta-Llama-3-8B",
        help="HuggingFace repo for the LLaMA 3 8B model",
    )
    parser.add_argument(
        "--output", default=None,
        help="Write results JSON to this file",
    )
    parser.add_argument(
        "--min-tokens", type=int, default=DEFAULT_MIN_TOKENS,
        help="Minimum distinct token IDs per (concept, language) group",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
        help="Min fraction for dominant feature value to be kept (0–1)",
    )
    args = parser.parse_args()

    data_dir      = Path(args.data_dir)
    languages     = SMOKE_LANGUAGES if args.smoke else list(TREEBANKS.keys())
    splits        = SMOKE_SPLITS    if args.smoke else FULL_SPLITS
    max_sentences = 200             if args.smoke else None

    print(f"Mode:      {'SMOKE' if args.smoke else 'FULL'}")
    print(f"Languages: {', '.join(languages)}")
    print(f"Splits:    {splits}")
    print(f"Model:     {args.model}")

    # 1. Load treebank data
    print("\n[1/4] Loading treebank data …")
    lang_words: dict[str, list] = {}
    for lang in languages:
        pairs = load_language(lang, splits, data_dir, max_sentences)
        lang_words[lang] = pairs
        print(f"  {lang}: {len(pairs):,} word-feature pairs")

    # 2. Load tokenizer
    print("\n[2/4] Loading tokenizer …")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"  Vocab size: {tokenizer.vocab_size:,}")

    # 3. Build token → feature map
    print("\n[3/4] Building token feature map (max-pooling) …")
    token_map = build_token_feature_map(
        tokenizer, lang_words,
        min_tokens=args.min_tokens,
        min_confidence=args.min_confidence,
    )
    feat_cats_found = sorted({fc for fc_map in token_map.values() for fc in fc_map})
    total_groups = sum(
        len(fv_map)
        for fc_map in token_map.values()
        for fv_map in fc_map.values()
    )
    print(f"  Feature categories: {', '.join(feat_cats_found)}")
    print(f"  Total (lang, feat_cat, feat_val) groups: {total_groups}")

    # 4. Load embeddings and L2-normalize
    print("\n[4/4] Loading token embedding matrix …")
    embeddings = load_embeddings(args.model)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms   # L2-normalize for cosine geometry
    print(f"  Embeddings normalised: shape {embeddings.shape}")

    # Analysis
    print("\nComputing cross-lingual cluster comparisons …")
    comparisons = analyze(token_map, embeddings)
    print(f"  {len(comparisons):,} pairwise group comparisons generated")

    print_results(comparisons)

    if args.output:
        out = {
            "config": {
                "model": args.model,
                "languages": languages,
                "splits": splits,
                "smoke": args.smoke,
                "min_tokens": args.min_tokens,
                "min_confidence": args.min_confidence,
            },
            "comparisons": comparisons,
        }
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
