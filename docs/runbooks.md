# Runbooks

## 1. Add transcripts for an existing ticker

### Prerequisites
- `.docx` transcript file for the earnings call
- Ticker already registered in `ticker_map.py` and `fiscal_calendar.py`

### Steps

**1. Place the file**

```
backend/transcripts/{TICKER}/{YYYY-MM-DD}.docx
```

The filename must be the **call date** (date the earnings call happened) in ISO format, not the quarter end date.

**2. Precompute chunks**

```bash
cd backend && .venv/bin/python3 scripts/precompute_chunks.py --tickers TICKER
```

Chunks the transcript, extracts a financial summary via LLM, and generates embeddings. Output saved to `backend/chunks/{TICKER}/{YYYY-MM-DD}.json`. Safe to rerun; add `--force` to reprocess an already-computed file.

**3. Write to DB**

```bash
cd backend && .venv/bin/python3 scripts/db_ingest_chunks.py --tickers TICKER
```

Upserts chunks into `document_chunks` and clears retrieval caches. Output should read:
```
[OK] TICKER/YYYY-MM-DD — N chunks + 1 financials chunk written
```

**4. Verify**

- Query the app: ask something about that company/quarter and confirm the new transcript is retrieved
- If on a live server, call `GET /warmup` to reprime caches after the first cold request

---

## 2. Add a new ticker (company)

### Prerequisites
- Know the company's **fiscal year end month** (the month their Q4 ends — e.g., January for companies like Salesforce, December for standard calendar-year companies). Look this up from their IR page or a known earnings date.
- Have at least one `.docx` transcript ready

### Steps

**1. Register the display name**

In [backend/app/rag/ticker_map.py](../backend/app/rag/ticker_map.py), add an entry (keep alphabetical order):

```python
"TICKER": "Company Name",
```

**2. Register the fiscal year end month**

In [backend/app/rag/fiscal_calendar.py](../backend/app/rag/fiscal_calendar.py), add the ticker under the correct month in `FY_END_MONTH`. If the company uses a standard December fiscal year end, add it under the `# December (12)` block. Getting this wrong causes quarters to be mislabeled in query results.

```python
"TICKER": 12,  # or whichever month
```

**3. Place transcript files and ingest**

Follow **Runbook 1** from step 1 onward. You can bulk-ingest all transcripts for the new ticker at once:

```bash
cd backend && .venv/bin/python3 scripts/precompute_chunks.py --tickers TICKER
cd backend && .venv/bin/python3 scripts/db_ingest_chunks.py --tickers TICKER
```

**4. Verify**

- Check that the ticker appears in company resolution by querying something like "What did [Company] say in their last earnings call?"
- Confirm the fiscal quarter label is correct (e.g., Q1 FY2025, not a mislabeled quarter)

---

## Common gotchas

| Issue | Cause | Fix |
|---|---|---|
| Quarter labels are wrong (e.g., Q1 shows as Q3) | Wrong `FY_END_MONTH` for the ticker | Correct `fiscal_calendar.py` and re-ingest with `--force` |
| `precompute_chunks` skips a file | `.json` already exists in `backend/chunks/` | Add `--force` flag |
| DB write fails with missing columns | Schema out of date | Run `alembic upgrade head` first |
| New ticker not found in queries | Cache not cleared | `db_ingest_chunks.py` clears it automatically; or restart the server |
