# Payment API

## Create payment intent

`POST /v1/payment-intents`

### Request fields

- `order_id` string, required
- `amount` integer, required
- `currency` string, required
- `payment_method_id` string, required
- `refund_mode` string, deprecated

### Notes

- `refund_mode` is deprecated and should not be sent by new clients.
- The service ignores `refund_mode` and does not use it to enable automatic refunds.
- Clients should rely on the refund workflow exposed through `POST /v1/refunds`.

## Create refund

`POST /v1/refunds`

### Request fields

- `payment_id` string, required
- `reason` string, required
- `requested_by` string, required

### Response

- `202 Accepted` when a refund request is queued for processing
