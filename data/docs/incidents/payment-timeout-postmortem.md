# Payment Timeout Postmortem

## Incident summary

Checkout payment attempts timed out for 14 minutes during a provider slowdown.

## Root causes

- The provider response time exceeded `PAYMENT_TIMEOUT_MS`
- Retries amplified load because clients and the service retried at the same time
- Operators manually replayed some requests before idempotency dashboards were updated

## Lessons

- Automatic refunds would have been risky during the incident because payment finality was unclear
- Retry coordination must improve before the team can safely automate more refund decisions

## Follow-up actions

- Tighten idempotency monitoring
- Add provider latency alerts
- Review timeout defaults per provider
