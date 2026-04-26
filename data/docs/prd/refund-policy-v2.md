# Refund Policy v2 PRD

## Summary

Refund Policy v2 updates the operating policy for refunds across checkout, payments, and support workflows.

## Current state

- Refund decisions are reviewed by operations for most payment failures
- Refund completion time is inconsistent across providers
- Agents often need engineering help to diagnose timeout-related failures

## Proposed policy

- Manual refunds remain the safe default for the MVP
- Automatic refunds are a future-state objective once payment reconciliation is more reliable
- Support tooling must show whether a refund was operator-triggered or system-triggered

## Constraints

- The policy should not assume that payment providers always return a final status within the checkout request window
- Accounting requires a clear audit trail for every refund
