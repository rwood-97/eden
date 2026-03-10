---
name: run-tests
description: Run the project test suite with coverage
---

Run the full test suite with:

```bash
uv run pytest --cov=src/eden tests/
```

If Chroma/RAG integration tests need to be skipped (no live index):

```bash
uv run pytest --cov=src/eden tests/ -k "not test_rag"
```

To run a specific test file:

```bash
uv run pytest tests/<file>.py -v
```

After running, check for failures before proceeding. If coverage drops, add tests to cover the new code.
