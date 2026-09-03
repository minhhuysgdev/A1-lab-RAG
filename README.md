# A1-lab-RAG

Multimodal Tiki search + RAG.

## Phase 0 — Data preparation

```bash
python scripts/prepare_data.py
```

Or open `notebooks/00_prepare_data.ipynb`.

### Output files

- `data/products_clean.jsonl` — cleaned corpus (~1.3k products)
- `data/preprocess_metadata.json` — stats
- `data/boilerplate_sentences.json` — removed boilerplate for manual review

### Record schema

Each line in `products_clean.jsonl` contains:

- `product_id`, `title`, `brand`, `category_l2`, `breadcrumbs`
- `text_segmented` (underscore tokens for BM25/PhoBERT)
- `text_natural` (spaces for bge-m3)
- `description_clean`
- `origin_norm`
- `attrs.color`, `attrs.size`, `attrs.variant` (each with `raw_value` + `value`)
- `image_urls_unique`
