# Release 1.2

## Checkout and payments

- Added the redesigned checkout shell for a limited merchant cohort
- Added admin-triggered refunds through the support backoffice
- Automatic refunds are not available in this release

## APIs

- Payment clients may still send `refund_mode`, but the field is deprecated
- Duplicate order protection now returns `409 Conflict` with error code `DUPLICATE_ORDER`

## Search

- Ranking v2 launched to 25 percent of traffic
