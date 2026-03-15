# Arpeggiator.ai — Technical Architecture & Process Flow

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Tech Stack](#2-tech-stack)
3. [Repository Structure](#3-repository-structure)
4. [Frontend Architecture](#4-frontend-architecture)
5. [Backend Architecture](#5-backend-architecture)
6. [ML Model & Generator System](#6-ml-model--generator-system)
7. [Music Generation Pipeline](#7-music-generation-pipeline)
8. [Authentication System](#8-authentication-system)
9. [API Reference](#9-api-reference)
10. [Process Flows](#10-process-flows)
11. [Configuration & Environment](#11-configuration--environment)
12. [Concurrency Model](#12-concurrency-model)

---

## 1. System Overview

Arpeggiator.ai is a full-stack web application that converts a free-text mood description into a downloadable MIDI arpeggio. Users describe how they want their music to feel (e.g. "dark and mysterious", "happy and energetic") and the system generates a unique MIDI pattern tuned to that emotion.

The application is split into two independently deployed services:

```
┌──────────────────────────────────┐        ┌──────────────────────────────────┐
│         React Frontend           │        │        FastAPI Backend           │
│         (Vite, port 3000)        │◄──────►│        (Uvicorn, port 8006)      │
│                                  │  HTTP  │                                  │
│  Landing → Login → Home/Studio   │  /api  │  Auth + Generation + Health      │
└──────────────────────────────────┘        └──────────────┬───────────────────┘
                                                           │
                                            ┌──────────────▼───────────────────┐
                                            │         PostgreSQL DB             │
                                            │    (users, sessions, tokens)     │
                                            └──────────────────────────────────┘
```

In development, Vite proxies all `/api` requests to `http://127.0.0.1:8006`, so the frontend only ever talks to its own origin. In production, `VITE_API_URL` is set to the deployed backend URL.

---

## 2. Tech Stack

### Frontend
| Layer | Technology |
|---|---|
| Framework | React 18 |
| Build tool | Vite |
| Routing | React Router v6 |
| Icons | Lucide React |
| Audio playback | Web Audio API (native browser) |
| Font | Inter (Google Fonts) |
| Styling | Plain CSS modules per component |

### Backend
| Layer | Technology |
|---|---|
| API framework | FastAPI |
| ASGI server | Uvicorn |
| ORM | SQLAlchemy |
| Database | PostgreSQL |
| Auth | JWT (python-jose) + bcrypt (passlib) |
| OAuth | Google OAuth 2.0 (popup flow) |
| ML runtime | PyTorch |
| MIDI generation | pretty_midi |
| Numerical ops | NumPy |
| Config management | Pydantic v2 + pydantic-settings |
| Concurrency | Starlette `run_in_threadpool` |

---

## 3. Repository Structure

```
Music-Arpeggiator/
│
├── src/                            # React frontend
│   ├── App.jsx                     # Root router
│   ├── landing-page.jsx            # Public landing page
│   ├── services/
│   │   └── api.js                  # Centralised HTTP client
│   └── components/
│       ├── Login.jsx / Login.css
│       ├── Signup.jsx / Signup.css
│       └── Home.jsx / Home.css     # Main studio page
│
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app + lifespan + CORS
│   │   ├── config.py               # Pydantic settings (reads .env)
│   │   ├── auth.py                 # JWT helpers + get_current_user
│   │   ├── database.py             # SQLAlchemy engine + session factory
│   │   │
│   │   ├── api/
│   │   │   ├── routes.py           # POST /generate-arpeggio
│   │   │   ├── schemas.py          # Pydantic request/response models
│   │   │   └── dependencies.py     # get_generator() FastAPI dependency
│   │   │
│   │   ├── generators/
│   │   │   ├── base.py             # BaseGenerator ABC + DTOs
│   │   │   ├── transformer.py      # CustomTransformerGenerator (active)
│   │   │   ├── pretrained_transformer.py  # PretrainedMusicTransformerGenerator
│   │   │   ├── mood_classifier.py  # MoodAlignmentScorer
│   │   │   └── mood_finetuner.py   # MoodFineTuner + FineTuningConfig
│   │   │
│   │   ├── model/
│   │   │   └── inference.py        # InferenceEngine + _MoodConditionedTransformer
│   │   │
│   │   ├── music/
│   │   │   ├── arpeggio_generator.py  # Note, Arpeggio, scale/key logic
│   │   │   ├── midi_renderer.py       # pretty_midi MIDI file rendering
│   │   │   └── tokenization.py        # REMI-style vocabulary + tokenizer
│   │   │
│   │   ├── routers/
│   │   │   └── auth.py             # /api/auth/* routes
│   │   ├── models/
│   │   │   └── user.py             # SQLAlchemy User model
│   │   └── services/               # Business logic helpers
│   │
│   ├── checkpoints/
│   │   ├── best_model.pt           # CustomTransformer weights (8 MB, active)
│   │   └── pretrained_music_transformer.pt  # REMI backbone (not yet trained)
│   │
│   └── scripts/
│       ├── train_mood_classifier.py
│       └── finetune_mood_adapter.py
│
├── run.sh                          # Dev launcher (starts both servers)
└── ARCHITECTURE.md                 # This document
```

---

## 4. Frontend Architecture

### Routing
Defined in `src/App.jsx` via React Router v6:

```
/          →  ArpeggiatorLanding   (public marketing page)
/login     →  Login                (JWT or Google auth)
/signup    →  Signup               (email/password registration)
/home      →  Home                 (protected studio page)
```

Every protected page checks `localStorage.getItem('token')` on mount and redirects to `/` if absent.

### API Client (`src/services/api.js`)
A single `request()` function handles all HTTP calls:
- Reads `VITE_API_URL` env var at build time; falls back to `/api` (Vite proxy)
- Injects `Authorization: Bearer <token>` header on every request
- Throws with `data.detail` from FastAPI on non-2xx responses

### Studio Page (`Home.jsx`)
The main page after login. Contains:

**Left panel — Build controls:**
- Tempo slider (20–400 BPM)
- Key selector (12 chromatic keys)
- Scale dropdown (16 scale types)
- Notes in Pattern slider (4–32 notes)
- Octave slider (0–8)
- Bars slider (1–8 repetitions)
- Mood text area (free-text input)

**Right panel — Preview & Export:**
- Bar chart visualisation of generated pitch sequence
- Real-time tracker (active note highlighted during playback via `requestAnimationFrame`)
- Instrument mode selector (4 timbres via Web Audio API oscillator types)
- Play / Stop button
- Export MIDI button

### Audio Playback
Entirely client-side using the **Web Audio API**:
1. Each note event from the API response is scheduled as an `OscillatorNode` + `GainNode` pair
2. Gain envelope: 10 ms attack → sustain → 30 ms release
3. A `requestAnimationFrame` loop compares `AudioContext.currentTime` against note positions to drive the tracker highlight
4. `AudioContext` is torn down and rebuilt on every Play press, guaranteeing playback always starts from beat 0

### MIDI Download
The API response includes a `midi_base64` field. The client decodes it with `atob()`, constructs a `Blob`, and triggers a download via a programmatically clicked `<a>` element.

---

## 5. Backend Architecture

### Application Startup (`main.py`)

FastAPI uses an `asynccontextmanager` lifespan hook that runs before the first request:

```
Application start
      │
      ├─► SQLAlchemy: Base.metadata.create_all()   (skipped if DB unreachable)
      │
      └─► _build_generator()
              │
              ├─ backbone checkpoint exists?
              │     YES → try PretrainedMusicTransformerGenerator.load()
              │     NO  → skip
              │
              ├─ pretrained failed or missing?
              │     → CustomTransformerGenerator.load()   ← always runs today
              │
              └─ set_generator(gen)   ← registers with FastAPI DI
```

If no checkpoint can be loaded, the app starts in **degraded mode**: health returns `status: "degraded"` and generation endpoints return HTTP 503.

### Middleware
- **CORS**: `CORSMiddleware` allows origins listed in `CORS_ORIGINS` (default: `localhost:3000`)
- No rate limiting or request size middleware currently

### Routers
| Prefix | File | Responsibility |
|---|---|---|
| `/api/auth/*` | `routers/auth.py` | Login, signup, Google OAuth, `/me` |
| `/api/generate-arpeggio` | `api/routes.py` | MIDI generation |
| `/health` | `main.py` | Service health check |
| `/` | `main.py` | Root info response |

### FastAPI Dependency Injection
`app/api/dependencies.py` maintains a module-level `_generator` variable:

```python
set_generator(gen)      # called once at startup
get_generator()         # injected into route handlers via Depends()
```

Route handlers never import the generator directly — they receive it through `Depends(get_generator)`, making the backend swappable without touching routes.

---

## 6. ML Model & Generator System

### Generator Hierarchy

```
BaseGenerator (ABC)
├── PretrainedMusicTransformerGenerator   ← REMI-style GPT (no checkpoint yet)
└── CustomTransformerGenerator            ← Active at runtime
        └── InferenceEngine
                └── _MoodConditionedTransformer (PyTorch)
```

### BaseGenerator Interface (`generators/base.py`)

All generators implement three abstract members:

```python
def load(self) -> None             # allocate weights; called once at startup
def generate(self, request: GenerationRequest) -> GenerationResult
@property is_ready: bool
@property name: str
```

**Transfer objects:**

`GenerationRequest` — backend-agnostic input:
```
key, scale, tempo, note_count, mood, octave, seed, pattern, bars
temperature, top_k, top_p, repetition_penalty, max_length  (per-request sampling overrides)
```

`GenerationResult` — output:
```
midi_bytes, notes: List[NoteResult], note_count, duration_seconds
key, scale, tempo, mood, pattern_used, sampling_params, alignment_score
```

### The Active Model: `_MoodConditionedTransformer`

Defined in `model/inference.py`. Architecture:

```
Input token IDs  ──► TokenEmbedding(vocab_size=411, d_model=128)
                  +
Positions        ──► PositionEmbedding(max_seq=512, d_model=128)
                  +
Mood label       ──► MoodEmbedding(num_moods=19, d_model=128)  [broadcast-added]
                        │
                        ▼
               TransformerDecoder (2 layers, 4 heads, ff_dim=512)
                        │
                        ▼
               Linear projection → logits (411)
```

Key architectural notes:
- **Decoder-only** with causal mask (GPT-style)
- **Mood injected additively** into every token position via `mood_embedding.unsqueeze(1)`
- **411-token vocabulary**: PAD/BOS/EOS/UNK + 19 MOOD + 128 NOTE_ON + 128 NOTE_OFF + 100 TIME_SHIFT + 32 VELOCITY
- Checkpoint: `best_model.pt` (8 MB), trained on mood-labelled MIDI data

### Vocabulary Layout

| Range | Tokens | Count |
|---|---|---|
| 0–3 | PAD, BOS, EOS, UNK | 4 |
| 4–22 | MOOD_melancholic … MOOD_ominous | 19 |
| 23–150 | NOTE_ON_0 … NOTE_ON_127 | 128 |
| 151–278 | NOTE_OFF_0 … NOTE_OFF_127 | 128 |
| 279–378 | TIME_SHIFT_1 … TIME_SHIFT_100 | 100 |
| 379–410 | VELOCITY_1 … VELOCITY_32 | 32 |

### 19 Valid Moods (label index order)

| Index | Mood | Index | Mood |
|---|---|---|---|
| 0 | melancholic | 10 | intense |
| 1 | dreamy | 11 | peaceful |
| 2 | energetic | 12 | dramatic |
| 3 | tense | 13 | epic |
| 4 | happy | 14 | mysterious |
| 5 | sad | 15 | romantic |
| 6 | calm | 16 | neutral |
| 7 | dark | 17 | flowing |
| 8 | joyful | 18 | ominous |
| 9 | uplifting | | |

---

## 7. Music Generation Pipeline

This is the full journey of a single generation request from user input to MIDI bytes.

### Step 1 — Mood Resolution

The free-text mood string is matched against the 19-label vocabulary:

```
"happy and energetic"
        │
        ▼
exact match in _VALID_MOODS?
    YES → label index used directly
    NO  → fuzzy match / cosine similarity against mood name embeddings
        → nearest label selected
```

Each mood has a corresponding `_MoodGenParams` profile that defines:
- `step_weights` — probability distribution over pitch intervals [-2, -1, +1, +2 scale degrees]
- `octave_jump_prob` — probability of a sudden register change
- `contour_bias` — tendency toward ascending (+1) or descending (-1) movement
- `start_region` — whether the sequence starts low, mid, or high in the register
- `rhythm_pattern` — beat-duration template (e.g. `[0.25, 0.25, 0.5, ...]`)
- `num_octaves` — register span

Example profiles:

| Mood | Contour | Octave Jump | Step Weights |
|---|---|---|---|
| energetic | ascending (+0.40) | 22% | heavy on +2 steps |
| melancholic | descending (−0.30) | 6% | heavy on −1, +1 steps |
| epic | ascending (+0.40) | 28% | heavy on +2 steps, wide span |
| calm | neutral | 8% | even distribution |

### Step 2 — Scale & Pitch Construction

```python
build_scale_pitches(key, scale, octave, num_octaves)
```

Maps the musical key (e.g. `C`) and scale type (e.g. `major`) to a list of valid MIDI pitches using `KEY_OFFSETS` and `SCALE_PATTERNS` lookup tables. The result is the set of legal pitches for the random walk.

Example: `C major, octave 4, 2 octaves` →
`[60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83]`

### Step 3 — Mood-Driven Pitch Walk

```python
generate_mood_pitch_sequence(scale_pitches, note_count, rng, **mood_params)
```

A stateful random walk through the scale degrees:

```
current_idx = start_region_index (low/mid/high based on mood profile)
for each note:
    1. Compute step weights adjusted by contour_bias
    2. Randomly select step delta from [-2, -1, +1, +2] using weighted choice
    3. Move current_idx by delta (clamped to scale bounds)
    4. With probability octave_jump_prob → shift by ±12 semitones
    5. Append pitch to sequence
```

This replaces the old teacher-forcing approach and always produces unique sequences because the RNG is seeded differently each call.

### Step 4 — Rhythm & Velocity Assignment (`_build_mood_notes`)

Each pitch is assigned a duration and velocity from the mood's `rhythm_pattern`:

```
rhythm_pattern = (0.25, 0.25, 0.5, 0.25, 0.25, 0.5)  # e.g. energetic
                  ↑ cycles through this template for all notes

velocity = base_velocity ± random_variation
         (base and variation defined per mood, e.g. energetic: 90 ± 20)
```

The result is a `List[Note]` where each `Note` has `pitch`, `velocity`, `position` (in beats), and `duration` (in beats).

### Step 5 — Pattern Tiling (Bars)

If `bars > 1`, the generated pattern is repeated end-to-end:

```python
pattern_duration = max(n.position + n.duration for n in notes)
for bar in range(1, bars):
    for note in original_notes:
        tiled_notes.append(Note(
            pitch=note.pitch,
            velocity=note.velocity,
            position=note.position + pattern_duration * bar,
            duration=note.duration,
        ))
```

A `bars=4` request with `note_count=16` produces 64 notes total in the MIDI file.

### Step 6 — MIDI Rendering

```python
MIDIRenderer.render(notes, tempo) → pretty_midi.PrettyMIDI
midi_to_bytes(midi_obj) → bytes
```

`pretty_midi` builds a standard MIDI file with:
- One instrument track (program 0, acoustic grand piano)
- Tempo set from the request
- Each `Note` mapped to `pretty_midi.Note(velocity, pitch, start_sec, end_sec)`

### Step 7 — Response Construction

```python
GenerationResult(
    midi_bytes=...,           # raw MIDI file bytes
    notes=[NoteResult(...)],  # individual note events for client-side audio
    note_count=...,
    duration_seconds=...,
    key, scale, tempo, mood, pattern_used,
    sampling_params={...},    # resolved temperature/top_k etc.
    alignment_score=None,     # None unless classifier checkpoint is loaded
)
```

The route handler base64-encodes `midi_bytes` and returns it alongside the note list in the JSON response.

---

## 8. Authentication System

### Password Auth Flow

```
POST /api/auth/signup
    body: { email, password, full_name }
        │
        ├─ bcrypt.hash(password)
        ├─ INSERT INTO users
        └─ return JWT access token + user object

POST /api/auth/login
    body: { email, password }
        │
        ├─ SELECT user WHERE email = ?
        ├─ bcrypt.verify(password, hash)
        └─ return JWT access token + user object
```

### JWT Structure

```
Header:  { alg: "HS256", typ: "JWT" }
Payload: { sub: user_email, exp: now + 30min }
Signature: HMAC-SHA256(header.payload, SECRET_KEY)
```

Tokens are stored in `localStorage` and sent as `Authorization: Bearer <token>` on every API request. The `get_current_user` dependency decodes the token and fetches the user from the DB.

### Google OAuth Flow

```
Client: window.open('/api/auth/google/login', popup)
        │
        ▼
Backend: redirect → Google consent screen
        │
Google: redirect → /api/auth/google/callback?code=...
        │
Backend: exchange code → Google access token
       → fetch user profile (email, name)
       → upsert user in DB
       → generate JWT
       → popup.postMessage({ type: 'GOOGLE_AUTH_SUCCESS', access_token, user })
        │
        ▼
Client: window.addEventListener('message') → store token → navigate('/home')
```

### Database Schema (User model)

```sql
CREATE TABLE users (
    id           SERIAL PRIMARY KEY,
    email        VARCHAR UNIQUE NOT NULL,
    full_name    VARCHAR,
    hashed_password VARCHAR,          -- NULL for OAuth-only accounts
    google_id    VARCHAR UNIQUE,      -- NULL for password accounts
    is_active    BOOLEAN DEFAULT TRUE,
    created_at   TIMESTAMP DEFAULT NOW()
);
```

---

## 9. API Reference

### `POST /api/generate-arpeggio`

**Request body:**
```json
{
  "key": "C",
  "scale": "major",
  "tempo": 120,
  "note_count": 16,
  "mood": "happy and energetic",
  "octave": 4,
  "bars": 2,
  "pattern": null,
  "seed": null,
  "temperature": null,
  "top_k": null,
  "top_p": null,
  "repetition_penalty": null,
  "max_length": null
}
```

**Response body:**
```json
{
  "midi_base64": "<base64-encoded MIDI file>",
  "notes": [
    { "pitch": 60, "velocity": 90, "position": 0.0, "duration": 0.5 },
    ...
  ],
  "key": "C",
  "scale": "major",
  "tempo": 120,
  "mood": "happy and energetic",
  "note_count": 32,
  "duration_seconds": 8.0,
  "sampling": {
    "temperature": 0.95,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "max_length": 1024
  },
  "alignment_score": null
}
```

**Auth required:** Yes (`Authorization: Bearer <token>`)

### `GET /health`
Returns generator status and resolved sampling config. No auth required.

### `POST /api/auth/login`
Returns `{ access_token, token_type, user }`.

### `POST /api/auth/signup`
Returns `{ access_token, token_type, user }`.

### `GET /api/auth/me`
Returns the current user profile. Auth required.

---

## 10. Process Flows

### Full Generation Flow (Request to Response)

```
Browser (Home.jsx)
    │
    │  POST /api/generate-arpeggio  { key, scale, tempo, note_count, mood, octave, bars }
    ▼
FastAPI route handler  (api/routes.py)
    │
    ├─ Pydantic validation (GenerateArpeggioRequest)
    │       - key in valid set?
    │       - scale in valid set?
    │       - tempo in [20, 400]?
    │       - note_count in [1, 128]?
    │       - mood length in [1, 200]?
    │
    ├─ Build GenerationRequest dataclass
    │
    └─ await run_in_threadpool(generator.generate, gen_request)
                    │
                    │  [worker thread — does not block event loop]
                    ▼
        CustomTransformerGenerator.generate()
                    │
                    ├─ 1. resolve_mood(mood_text) → label_index (0-18)
                    │
                    ├─ 2. build_scale_pitches(key, scale, octave) → [MIDI pitches]
                    │
                    ├─ 3. generate_mood_pitch_sequence(pitches, note_count, rng, **mood_params)
                    │        → List[pitch_int]
                    │
                    ├─ 4. _build_mood_notes(pitches, tempo, mood_params)
                    │        → List[Note(pitch, velocity, position, duration)]
                    │
                    ├─ 5. tile notes if bars > 1
                    │        → List[Note]  (bars × note_count notes)
                    │
                    ├─ 6. MIDIRenderer.render(notes, tempo)
                    │        → pretty_midi.PrettyMIDI
                    │
                    └─ 7. midi_to_bytes() → GenerationResult
                    │
    ┌───────────────┘
    │
    ├─ Build SamplingParams (fill defaults from settings if generator omitted them)
    ├─ base64-encode midi_bytes
    └─ return GenerateArpeggioResponse JSON
                    │
                    ▼
Browser
    ├─ setResult(data)                          → re-renders bar chart
    ├─ handlePlay() → schedules Web Audio nodes → plays notes
    └─ handleDownload() → decodes base64 → saves .mid file
```

### Authentication Flow (Email/Password)

```
Browser                    FastAPI                    PostgreSQL
   │                          │                           │
   │  POST /api/auth/signup   │                           │
   │─────────────────────────►│                           │
   │  { email, pw, name }     │                           │
   │                          ├─ bcrypt.hash(pw)          │
   │                          │                           │
   │                          │  INSERT INTO users        │
   │                          │──────────────────────────►│
   │                          │◄──────────────────────────│
   │                          │                           │
   │                          ├─ create_access_token()    │
   │◄─────────────────────────│                           │
   │  { access_token, user }  │                           │
   │                          │                           │
   │  localStorage.setItem()  │                           │
   │  navigate('/home')       │                           │
```

### Playback & Tracker Flow (Client-side only)

```
handlePlay() called
    │
    ├─ stopPlayback()                   ← tear down any existing AudioContext
    ├─ new AudioContext()
    ├─ schedule all oscillators          ← one per note, future-dated
    ├─ setIsPlaying(true)
    │
    └─ requestAnimationFrame(tick)
            │
            ├─ every frame:
            │    elapsedBeats = (ctx.currentTime - startTime) / secondsPerBeat
            │    find displayNote where position ≤ elapsedBeats < position + duration
            │    setActiveNoteIdx(i)  →  re-render highlights active bar
            │
            └─ continues until stopPlayback() is called
                    │
                    (either: Stop button clicked, or setTimeout after duration_seconds fires)
```

---

## 11. Configuration & Environment

All settings are managed by `app/config.py` using Pydantic's `BaseSettings`. Values are read from environment variables or a `.env` file.

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://...@127.0.0.1:5432/arpeggiator` | PostgreSQL connection string |
| `SECRET_KEY` | *(change in prod)* | JWT signing key |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | JWT TTL |
| `GOOGLE_CLIENT_ID` | `""` | Google OAuth app ID |
| `GOOGLE_CLIENT_SECRET` | `""` | Google OAuth secret |
| `GOOGLE_REDIRECT_URI` | `http://localhost:8006/api/auth/google/callback` | OAuth callback |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed CORS origins |
| `PRETRAINED_CHECKPOINT` | `checkpoints/pretrained_music_transformer.pt` | REMI backbone path |
| `CUSTOM_CHECKPOINT` | `checkpoints/best_model.pt` | Fallback model path |
| `MOOD_ADAPTER_CHECKPOINT` | `checkpoints/mood_adapter.pt` | Optional adapter |
| `MOOD_CLASSIFIER_CHECKPOINT` | `checkpoints/mood_classifier.pt` | Optional classifier |
| `GENERATION_TEMPERATURE` | `0.95` | Sampling temperature |
| `GENERATION_TOP_K` | `50` | Top-k filter (0 = disabled) |
| `GENERATION_MAX_GEN_TOKENS` | `1024` | Max tokens per request |
| `ALIGNMENT_SCORE_THRESHOLD` | `0.0` | Min classifier score to accept (0 = no retry) |
| `ALIGNMENT_MAX_ATTEMPTS` | `3` | Max retry attempts |
| `VITE_API_URL` | *(unset — falls back to `/api`)* | Frontend: backend base URL |

---

## 12. Concurrency Model

### Why Non-Blocking Inference Matters

FastAPI runs on a single-threaded asyncio event loop. If `generator.generate()` — which runs PyTorch model inference — were called directly inside an `async def` handler, it would block the event loop for the full duration of the model forward pass (potentially hundreds of milliseconds), making the server unable to handle any other requests in that window.

### Solution: `run_in_threadpool`

```python
result = await run_in_threadpool(generator.generate, gen_request)
```

Starlette's `run_in_threadpool` submits the blocking call to a thread pool executor. The event loop suspends the current coroutine and remains free to process other requests until the worker thread completes.

### Thread Safety

`InferenceEngine` and `CustomTransformerGenerator` both use a `threading.RLock` to guard all model forward passes:

```python
with self._lock:
    with torch.no_grad():
        logits = self._model(src_in, mood_t)
```

This allows multiple concurrent requests to share a single model instance safely — each request queues for the lock, performs inference, and releases it.

### Database Sessions

Each request gets its own SQLAlchemy session via the `get_db` generator dependency. Sessions are always closed in a `finally` block, preventing connection leaks even when exceptions occur.
