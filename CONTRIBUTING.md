# Contributing to TrueWatch AI

## Before you start

- Use Node.js 20 or newer.
- Run `npm run setup` once to install dependencies and create your local config.
- Never add real credentials, camera information, or personal data to the repository.

## Making a change

1. Keep each change focused on one feature or fix.
2. Put browser-facing work in `frontend/` and API work in `backend/`.
3. Follow the existing feature structure rather than adding cross-cutting code
   to a route or controller.
4. Run `npm run check` before requesting review.

## Code boundaries

- UI components use service modules; they should not embed API endpoints.
- API routes stay thin, controllers handle HTTP details, and services own the
  application behaviour.
- Add placeholder configuration only to `.env.example`; keep actual values in
  untracked `.env` files.

## Pull requests

Describe the user-visible change, note any API or configuration changes, and
include a screenshot for visual frontend work when practical.
