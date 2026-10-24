# AI-Based Customer Support Ticketing System

Customers open tickets; a scikit-learn model classifies each one (category + priority), routes it to the right team, and assigns the least-loaded agent. SLA deadlines are tracked by a background job, and dashboards show agent performance and customer satisfaction (CSAT).

| Layer | Tech |
|---|---|
| Frontend | Next.js 14 (App Router), TypeScript, Tailwind CSS, Recharts |
| Backend API | Node.js 20, Express, TypeScript, Mongoose, node-cron, JWT |
| ML service | Python 3.11, FastAPI, scikit-learn (TF-IDF + LinearSVC), joblib |
| Database | MongoDB 7 |

## Prerequisites

- **Docker** with Compose v2 (`docker compose version`), and free ports **3000**, **4000**, **8000**.
- Nothing else. Node, Python and MongoDB all run in containers.

## Run it (one command)

```bash
docker compose up --build
```

First build takes a few minutes (npm/pip installs, model training). When the logs show `[api] listening on :4000`:

- App: <http://localhost:3000>
- API: <http://localhost:4000/api/health>
- ML service docs: <http://localhost:8000/docs> (metrics at `/metrics`)

The backend seeds demo data automatically on first start (`AUTO_SEED=true`, only when the database is empty).

### Demo accounts (password for all: `password123`)

| Role | Email | Notes |
|---|---|---|
| Admin | `admin@example.com` | Sees everything, can reassign tickets |
| Agent | `priya@example.com` | Billing team. Others: `marcus@` (billing), `sofia@`, `liam@` (technical), `noah@` (account), `emma@` (general) |
| Customer | `ava@example.com` | Others: `ben@`, `chloe@` (all `@example.com`) |

Seed data: 6 agents, 3 customers, 200 tickets over 90 days (about 15 created in the last 30 hours so live SLA countdowns are visible), and CSAT responses on roughly 72% of resolved tickets.

### Try the main flows

1. Sign in as `ava@example.com` and create a ticket, e.g. *"Charged twice"* / *"Please respond immediately, this is urgent."* You'll see the category, priority, confidence and assigned agent.
2. Sign in as that agent and press **Mark resolved**. The CSAT link is printed in the backend logs (`docker compose logs backend | grep CSAT`) as a stand-in for an email.
3. Open the link (`/feedback/<token>`) and rate the ticket. It appears under **Customer satisfaction** as an admin.
4. Stop the ML service (`docker compose stop ml-service`) and create another ticket: it is still routed, using keyword rules, and the backend logs a warning.

### Reset or reseed

```bash
docker compose down -v          # wipes the MongoDB volume; next `up` reseeds
docker compose exec backend node dist/seed.js   # force a reseed without wiping the volume
```

## Architecture

```
                    ┌────────────────────────────┐
   Browser  ───────▶│  Next.js frontend  :3000   │
      │             └────────────────────────────┘
      │  REST + JWT (NEXT_PUBLIC_API_URL)
      ▼
┌──────────────────────────────┐   POST /classify    ┌──────────────────────────┐
│  Express API  :4000          │────────────────────▶│  FastAPI ML service :8000│
│  • auth / RBAC middleware    │◀────────────────────│  TF-IDF + LinearSVC      │
│  • ticket routing & assign   │  {category,priority,│  (category + priority)   │
│  • keyword-rule fallback     │   confidence,team}  └──────────────────────────┘
│  • node-cron SLA sweep (60s) │        │ timeout / error
│  • analytics aggregations    │        └─▶ fallback: keyword rules + warning log
└──────────────┬───────────────┘
               │ Mongoose
               ▼
        ┌──────────────┐
        │  MongoDB :27017 │  users · tickets · feedbacks
        └──────────────┘
```

**Ticket creation flow:** `POST /api/tickets` → call ML (3 s timeout) or fall back to rules → pick the least-loaded agent in the suggested team → store `sla_due_at = now + SLA(priority)` → return ticket plus routing details.

**Teams:** billing + refund → `billing`; technical → `technical`; account → `account`; feature_request + other → `general`.

### SLA tracking

| Priority | Window |
|---|---|
| urgent | 1 h |
| high | 4 h |
| medium | 24 h |
| low | 72 h |

`slaDueAt` is stored at creation. A `node-cron` job runs every minute (and once at startup) and sets `slaStatus` on open/in-progress tickets: `at_risk` once 80% of the createdAt→slaDueAt window has elapsed, `breached` after the due time. On resolution, `slaMet` is recorded. The UI badge counts down live in the browser using the same 80% rule, so it never lags behind the job.

## API reference

All routes are under the Express API (`:4000`). "Auth" is a `Authorization: Bearer <jwt>` header.

| Method | Path | Auth / roles | Description |
|---|---|---|---|
| GET | `/api/health` | none | Liveness check |
| POST | `/api/auth/login` | none | `{email, password}` → `{token, user}` |
| POST | `/api/auth/register` | none | Create a **customer** account |
| GET | `/api/auth/me` | any | Current user |
| GET | `/api/tickets` | any (row-filtered) | List; filters `status` (`active`/…), `priority`, `category`, `slaStatus`, `search`, `sort` (`newest`/`sla`), `page`, `limit` |
| POST | `/api/tickets` | customer | `{subject, body}` → classify, route, assign; returns `{ticket, routing}` |
| GET | `/api/tickets/:id` | any (row-filtered) | Ticket detail |
| PATCH | `/api/tickets/:id/status` | agent, admin | `{status}`; resolving stamps `resolvedAt`, `slaMet`, and creates the CSAT token |
| PATCH | `/api/tickets/:id/assign` | admin | `{agentId}` |
| GET | `/api/agents` | admin | Agents with current open load |
| GET | `/api/analytics/agents?days=90` | admin, agent (own data) | Per-agent resolved, avg resolution, SLA %, open load, weekly trend |
| GET | `/api/analytics/csat?days=90` | admin, agent (own data) | Avg, distribution, weekly trend, by category, response rate |
| GET | `/api/feedback/:token` | public (token) | Survey info |
| POST | `/api/feedback/:token` | public (token) | `{rating: 1-5, comment?}`; one submission per ticket |

ML service (`:8000`):

| Method | Path | Description |
|---|---|---|
| POST | `/classify` | `{subject, body}` → `{category, priority, confidence, priority_confidence, suggested_agent_team}` |
| GET | `/health` | Liveness + model loaded |
| GET | `/metrics` | `metrics.json` written by the last training run |

**Visibility rules:** customers see only their own tickets; agents see tickets assigned to them plus unassigned tickets in their team's queue; admins see everything.

## How the ML classifier works

- **Data:** `ml-service/data/tickets.csv`, 300 synthetic tickets (50 per category: billing, technical, account, refund, feature_request, other) with a priority label each. Generated deterministically by `app/dataset.py` from phrase templates. To make evaluation less trivial, 30% of subjects are generic ("Need help"), 30% of bodies borrow a sentence from another category (often the confusable partner, e.g. billing↔refund), and 10% of priority cues are off by one level.
- **Features:** TF-IDF over `subject. subject. body` (subject repeated to up-weight it), unigrams + bigrams, sublinear term frequency.
- **Models:** two independent `LinearSVC(class_weight="balanced")` pipelines, one for category and one for priority.
- **Confidence:** LinearSVC outputs margins, not probabilities. We apply a softmax to the scaled margins (`MARGIN_SCALE = 4`). It ranks predictions sensibly but is **not calibrated**.
- **Split & persistence:** stratified 80/20 split (240 train / 60 test, `random_state=42`). The shipped model is trained on the 80% split so the reported numbers describe the shipped artifact. `python train.py --refit-full` refits on all 300 rows. The bundle is saved with joblib to `models/model.joblib` and `metrics.json`.

### Accuracy on the bundled test split (60 tickets)

Measured with scikit-learn 1.8 during development:

| Target | Test accuracy | Macro-F1 | 5-fold CV accuracy (train split) |
|---|---|---|---|
| Category (6 classes) | **86.7%** | 0.866 | 93.3% ± 1.6 |
| Priority (4 classes) | **93.3%** | 0.940 | 91.3% ± 3.6 |

Per-category F1 on the test split: other 1.00, billing 0.95, feature_request 0.90, refund 0.84, account 0.80, technical 0.70.

> **Read these numbers with care.** The data is synthetic and template-based, the test set is only 60 rows, and a gap between test accuracy and CV is expected noise at this size. Real tickets are messier; expect lower accuracy until you retrain on real, labeled history. Docker pins scikit-learn 1.5.2 and retrains at image build, so the figures your build prints (and `GET :8000/metrics` returns) may differ slightly.

### Retraining

```bash
cd ml-service
pip install -r requirements.txt
python train.py --regenerate     # rebuild the CSV and retrain
python train.py                  # retrain on the existing CSV (swap in your own labeled data)
```

CSV columns: `subject, body, category, priority`. Rebuild the image to ship a new model: `docker compose build ml-service`.

### Fallback routing

If the ML call times out (`ML_TIMEOUT_MS`), errors, or returns an invalid payload, the backend logs `[classifier] ML service unavailable (...)` and uses keyword rules (`backend/src/services/classifier.ts`). Tickets from this path are marked `routedBy: "rules"` and shown that way in the UI.

## Configuration

Each service has a `.env.example` with working defaults. Docker Compose sets the same values inline, so no `.env` file is needed for `docker compose up`.

| Service | Variable | Default | Purpose |
|---|---|---|---|
| backend | `PORT` | `4000` | API port |
| backend | `MONGO_URI` | `mongodb://mongo:27017/ticketing` | Mongo connection (use `localhost` outside Docker) |
| backend | `JWT_SECRET` | dev value | **Change for anything beyond local use** |
| backend | `JWT_EXPIRES_IN` | `8h` | Token lifetime |
| backend | `ML_SERVICE_URL` | `http://ml-service:8000` | ML base URL |
| backend | `ML_TIMEOUT_MS` | `3000` | Timeout before fallback |
| backend | `CORS_ORIGIN` | `http://localhost:3000` | Comma-separated allowed origins |
| backend | `AUTO_SEED` | `true` | Seed when the DB has no users |
| frontend | `NEXT_PUBLIC_API_URL` | `http://localhost:4000` | API URL **as seen by the browser**; baked in at build time |
| ml-service | `MODEL_PATH` | `models/model.joblib` | Model bundle location |
| ml-service | `PORT` | `8000` | Uvicorn port |

## Running without Docker (development)

```bash
# 1. MongoDB on localhost:27017 (e.g. docker run -p 27017:27017 mongo:7)

# 2. ML service
cd ml-service && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python train.py
uvicorn app.main:app --port 8000

# 3. Backend (new terminal)
cd backend && cp .env.example .env
# edit .env: MONGO_URI=mongodb://localhost:27017/ticketing, ML_SERVICE_URL=http://localhost:8000
npm install && npm run seed && npm run dev

# 4. Frontend (new terminal)
cd frontend && cp .env.example .env.local && npm install && npm run dev
```

## Project layout

```
.
├── docker-compose.yml
├── frontend/      Next.js app (src/app routes, src/components, src/lib)
├── backend/       Express API (src/routes, services, models, seed.ts)
├── ml-service/    FastAPI app, train.py, data/tickets.csv
└── docs/screenshots/
```

## Screenshots

Add your own captures to `docs/screenshots/`:

| View | File |
|---|---|
| Ticket list with live SLA badges | `![Ticket list](docs/screenshots/tickets.png)` |
| Ticket detail | `![Ticket detail](docs/screenshots/ticket-detail.png)` |
| Agent performance dashboard | `![Agent dashboard](docs/screenshots/agents.png)` |
| CSAT analytics | `![CSAT dashboard](docs/screenshots/csat.png)` |
| Feedback form | `![Feedback form](docs/screenshots/feedback.png)` |

## Limitations / next steps

- **Synthetic training data.** Accuracy figures are not predictive of production. Next: train on real labeled tickets, add calibration (e.g. `CalibratedClassifierCV`), and route low-confidence tickets to a triage queue.
- **No verified end-to-end run in the authoring environment.** The ML pipeline was executed and measured; the TypeScript services and Docker builds were written and cross-checked (env names, ports, routes) but not compiled or started there. If a build fails, run `npm run typecheck` in `frontend/` and `backend/` first.
- **No lockfiles.** `npm install` resolves fresh versions; commit `package-lock.json` after the first install for reproducible builds.
- **CSAT email is simulated.** The feedback link is logged to the backend console. Wire up an email provider (SES, Postmark, SMTP) in `routes/tickets.ts`.
- **Shared types are duplicated.** `backend/src/types/shared.ts` is the source of truth and `frontend/src/lib/types.ts` mirrors it. Promote to an npm workspace package to remove the copy.
- **Auth hardening.** JWT is stored in `localStorage` with no refresh tokens, rate limiting, password reset, or account lockout. Prefer httpOnly cookies and add `helmet` + rate limits before exposing publicly.
- **Ticket workflow is minimal.** No replies/comments thread, attachments, reopening by customers, business-hours SLA calendars, or pause-on-customer-reply.
- **SLA granularity.** The sweep runs once a minute (the finest `node-cron` interval); the UI countdown is computed client-side for real-time feel.
- **Seeding wipes data.** `npm run seed` / `seed.js` deletes all users, tickets and feedback first.
- **Tests.** No automated test suite yet. Good first targets: rules fallback, SLA sweep thresholds, role visibility filters, analytics aggregations.

## License

MIT, see [LICENSE](LICENSE).
