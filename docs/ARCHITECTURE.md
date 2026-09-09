# Architecture

TrueWatch AI is a small JavaScript workspace with two deployable applications:
the browser client in `frontend/` and the HTTP API in `backend/`.

```text
Browser
  │
  ▼
React + Vite frontend
  │  /api/* (development proxy)
  ▼
Express API
  │
  ├── controllers   translate HTTP requests and responses
  ├── services      implement the domain behaviour
  └── data          provides demo fixtures today
```

## Frontend

The frontend keeps page assembly in `src/routes/` and feature-level building
blocks in `src/components/`. Pages obtain data through `src/services/`, not by
calling `fetch` directly. That keeps UI code independent of whether the API is
backed by mock data, a local appliance, or a hosted service.

`src/hooks/useConsole.js` owns the console's async state: polling, searching,
alert actions, and evidence verification. It is the natural place to introduce
a WebSocket or server-sent-events stream later.

Canvas animations live in `src/canvas/`. Each scene owns its lifecycle and is
mounted by a UI component, keeping animation code out of route components.

## Backend

Each API resource follows the same path:

```text
route → controller → service → data source
```

- **Routes** declare endpoints and connect middleware.
- **Controllers** validate/shape HTTP input and output.
- **Services** hold domain operations such as alert simulation or footage search.
- **Data** contains replaceable mock fixtures for the demo.

When a persistent store is introduced, add repositories or a `db/` layer under
`backend/src/` and have services depend on that layer. Keep controllers unaware
of database and model-provider details.

## Configuration

Each workspace has an `.env.example` file committed to the repository. The root
`npm run setup` command makes a local copy only if a `.env` is absent. Real
credentials and deployment-specific values must remain in untracked `.env`
files.

## Production considerations

This is a prototype, so its current authentication, evidence verification, and
data stores are demonstrations only. Before a deployment, add verified identity
and authorization, durable audit storage, input validation, rate limiting,
observability, and an automated test suite.
