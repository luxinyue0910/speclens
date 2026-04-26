# Order API

## Create order

`POST /v1/orders`

### Request fields

- `cart_id` string, required
- `checkout_version` string, optional
- `customer_email` string, required

### Error behavior

- `409 Conflict` is returned for a duplicate order submission.
- The error code for duplicate order is `DUPLICATE_ORDER`.

## Get order

`GET /v1/orders/{order_id}`

### Response fields

- `status`
- `payment_status`
- `fulfillment_status`
