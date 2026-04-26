# Search Ranking v2 PRD

## Objective

Search Ranking v2 improves relevance for long-tail queries and low-data catalogs.

## Planned changes

- Increase the lexical candidate set from 100 to 400 items
- Add semantic relevance features from the embedding service
- Add popularity and inventory freshness signals to the ranking model

## Expected tradeoffs

- Better relevance on ambiguous queries
- Higher ranking latency due to the larger candidate set and additional feature fetches
- More operational sensitivity to feature store cache misses

## Success criteria

- 6 percent improvement in purchase-through-search
- No more than 15 percent increase in p95 search latency
