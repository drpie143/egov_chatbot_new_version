# API

## GET /health

Reports resource and model availability.

## POST /chat

Accepts `question` and optional `session_id`; returns answer, source cards, request id, latency, cache flag, and context source.

## GET /search

Query params:

- `q`: search query
- `limit`: maximum number of results

Returns procedure cards with title, URL, agency, score, and snippet.

## POST /feedback

Stores a rating: `like`, `dislike`, or `neutral`.

## POST /clear_session

Clears the current in-memory conversation context.

## Legacy Aliases

- `POST /save_feedback`
- `POST /update_popular`

