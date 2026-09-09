# TrueWatch AI

An edge-first video analytics prototype for monitoring remote border areas. The
project pairs a responsive operator console with an Express API, using isolated
mock data so that real cameras, models, and storage can be integrated
incrementally.

> **Project status:** active prototype. Authentication, persistence, and video
> intelligence are represented by safe demo flows and mock services.

## Highlights

- Public product site, sign-in/sign-up journeys, and an operator console
- Live-wall simulation, sector map, alerts, evidence chain, and footage search
- React frontend with a resource-oriented Express API
- Clear service boundaries for replacing demo data with production systems
- Reduced-motion support and a responsive visual system

## Quick start

**Prerequisite:** Node.js 20 or newer.

```bash
npm run setup
npm run dev
```

The setup command installs workspace dependencies and creates local environment
files from their examples when needed. It never overwrites an existing `.env`.

| Service | Address |
| --- | --- |
| Web app | http://localhost:5173 |
| API | http://localhost:4000/api |
| API health check | http://localhost:4000/api/health |

Useful commands:

```bash
npm run dev:web    # Run only the frontend
npm run dev:api    # Run only the API
npm run build      # Build the frontend for production
```

## Project layout

```text
.
├── frontend/                  React + Vite application
│   ├── src/
│   │   ├── components/        Reusable UI, grouped by feature
│   │   ├── routes/            Page-level route components
│   │   ├── canvas/            Animated visual scenes
│   │   ├── hooks/             Shared UI and console state
│   │   ├── services/          API client and resource adapters
│   │   ├── context/           App-wide UI state
│   │   ├── data/              Presentation copy and offline fallbacks
│   │   ├── styles/            Global styles and keyframes
│   │   └── utils/             Small presentation helpers
│   └── public/                Static browser assets
├── backend/                   Express API
│   └── src/
│       ├── routes/            HTTP endpoints
│       ├── controllers/       Request and response handling
│       ├── services/          Domain logic and mock implementations
│       ├── data/              Demo fixtures
│       ├── middleware/        Error and not-found handling
│       ├── config/            Environment configuration
│       ├── app.js             Application composition
│       └── server.js          Server entry point
├── docs/                      Architecture and contributor notes
├── scripts/                   Local development helpers
├── package.json               Workspace commands
└── .gitignore                 Shared repository exclusions
```

For how the frontend and API fit together, see [the architecture guide](docs/ARCHITECTURE.md).

## Application routes

| Route | Purpose |
| --- | --- |
| `/` | Public overview and product story |
| `/login` | Operator sign-in flow |
| `/signup` | Account-request flow |
| `/console` | Video operations console |

## API surface

Responses use `{ "data": ... }`; failures use `{ "error": "..." }`.

| Area | Endpoints |
| --- | --- |
| Health | `GET /api/health` |
| Authentication | `POST /api/auth/login`, `POST /api/auth/register`, `GET /api/auth/units`, `GET /api/auth/roles` |
| Cameras | `GET /api/cameras`, `GET /api/cameras/posts` |
| Alerts | `GET /api/alerts`, `POST /api/alerts/simulate`, `POST /api/alerts/:id/act`, `POST /api/alerts/:id/dismiss` |
| Search | `GET /api/search/placeholders`, `POST /api/search` |
| Evidence | `GET /api/evidence/chain`, `POST /api/evidence/verify` |
| Analytics | `GET /api/analytics`, `GET /api/analytics/traffic` |

## Next integration points

The application is designed so the demo can become a real system one layer at a
time:

- Replace `backend/src/services/` mock implementations with real camera,
  database, queue, and model integrations.
- Add token verification in `backend/src/middleware/` before exposing protected
  routes.
- Replace client polling in `frontend/src/hooks/useConsole.js` with a WebSocket
  or server-sent-events feed when live updates are available.
- Connect the evidence service to durable hashes and object storage before
  treating it as an auditable chain.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a change. Keep
secrets in local `.env` files and commit only the included `.env.example`
templates.
