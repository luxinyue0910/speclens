# Payment Service Design

## Scope

This document defines the MVP payment service for authorization, capture, and refunds.

## Endpoints

- `POST /v1/payment-intents` creates a payment intent
- `POST /v1/refunds` creates a refund
- `GET /v1/payments/{payment_id}` fetches payment status

## MVP constraints

- The MVP does not support automatic refunds.
- Refunds are admin-triggered only through the support backoffice workflow.
- Checkout can display refund policy language, but it cannot directly trigger automatic refunds after downstream failures.

## Design rationale

- Provider responses are occasionally delayed beyond the checkout request timeout.
- Automatic refunds would have increased the risk of duplicate or conflicting refund attempts.
- Manual review keeps accounting and support workflows aligned during the MVP phase.

## Configuration

- `PAYMENT_TIMEOUT_MS` defaults to `3000`
- `PAYMENT_RETRY_LIMIT` defaults to `2`
- `PAYMENT_PROVIDER_BASE_URL` points to the active provider environment

## API notes

- The service accepts an idempotency key for payment-intent creation.
- The legacy `refund_mode` field is deprecated and ignored by the service.
