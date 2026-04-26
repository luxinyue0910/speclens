# Notification System Design

## Summary

The notification platform delivers order, payment, and shipment events to internal and external consumers.

## Architecture

- Event producers write notification jobs to a queue
- Workers process jobs asynchronously
- Retries are handled outside the request path

## Tradeoffs

- Asynchronous delivery reduces user-facing latency in checkout and order APIs.
- The system accepts eventual consistency between the write API and downstream notifications.
- Debugging is slightly harder because delivery failures are decoupled from the original request.

## Why async delivery

- The team wanted to remove third-party webhook latency from the user request path.
- Sync notification delivery caused timeout spikes during provider incidents.
- Async workers make retry policy explicit and configurable.

## Configuration

- `NOTIFICATION_WORKER_CONCURRENCY` defaults to `12`
- `NOTIFICATION_MAX_RETRIES` defaults to `5`
- `NOTIFICATION_DEAD_LETTER_TOPIC` stores permanently failed jobs
