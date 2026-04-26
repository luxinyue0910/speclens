# Search Latency Postmortem

## Incident summary

After Search Ranking v2 was enabled for 100 percent of traffic, p95 search latency increased from 420 ms to 910 ms.

## Root causes

- The candidate set increased from 100 to 400 items before ranking.
- The ranker fetched more popularity and semantic features per query.
- Feature store cache misses caused repeated network calls during high traffic periods.
- Early termination logic from Ranking v1 was removed during the rollout.

## Contributing factors

- The rollout guardrail only checked average latency, not p95.
- The feature service cache was undersized for the larger candidate set.

## Mitigations

- Restore early termination for low-confidence candidates
- Reduce cache miss rate in the feature service
- Add a canary check for p95 search latency before full rollout
