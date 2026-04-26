# Notification API

## Send notification

`POST /v1/notifications`

### Request fields

- `event_type` string, required
- `recipient` string, required
- `template_id` string, required
- `delivery_mode` string, optional
- `sync_delivery` boolean, deprecated

### Notes

- `sync_delivery` is deprecated because the platform uses asynchronous workers by default.
- `delivery_mode=async` is the recommended default for all new integrations.

## Response

- `202 Accepted` indicates the job was accepted for async processing
