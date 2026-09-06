# 🧠 AI Long Term Memory for Home Assistant

[![Home Assistant](https://img.shields.io/badge/home%20assistant-%2341BDF5.svg?style=for-the-badge&logo=home-assistant&logoColor=white)](https://home-assistant.io)
[![hacs](https://img.shields.io/badge/HACS-Custom-orange.svg?style=for-the-badge)](https://www.hacs.xyz/docs/faq/custom_repositories/)
[![License][license-shield]](LICENSE.md)

[license-shield]: https://img.shields.io/github/license/Riscue/ha-ai-memory.svg?style=for-the-badge

[![Active installations](https://img.shields.io/badge/dynamic/json?style=for-the-badge&color=41BDF5&logo=home-assistant&label=active%20installations&cacheSeconds=15600&url=https://analytics.home-assistant.io/custom_integrations.json&query=$.ai_memory.total)](https://github.com/Riscue/ha-ai-memory)
[![GitHub Downloads (all assets, latest release)](https://img.shields.io/github/downloads/Riscue/ha-ai-memory/latest/total?label=downloads&style=for-the-badge)](https://github.com/Riscue/ha-ai-memory/releases)

[![GitHub Release](https://img.shields.io/github/release/Riscue/ha-ai-memory.svg?style=for-the-badge)](https://github.com/Riscue/ha-ai-memory/releases)
[![GitHub Activity](https://img.shields.io/github/commit-activity/y/Riscue/ha-ai-memory.svg?style=for-the-badge)](https://github.com/Riscue/ha-ai-memory/commits/master)

![Icon](custom_components/ai_memory/brand/logo.png)

[![Add Integration](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=ai_memory)

A long-term, semantic memory system for Home Assistant's Assist LLM agents. Store facts, preferences,
household context, and automation history that persists across conversations — and let your agents
recall them through vector similarity search.

- **Native LLM tools**: `add_memory`, `search_memory`, and `delete_memory` are automatically exposed
  to every Assist agent.
- **HA services**: `add_memory`, `list_memories`, `search_memory`, `delete_memory` services for
  automations and scripts, all with `supports_response=OPTIONAL` so you can capture the result into
  a script response variable.
- **Memory Palace**: Memories are organized into **Wing → Room** (e.g. `household/devices`,
  `personal/preferences`) so the agent — or the user — can scope recall and writes.
- **Layered memory (L0–L3)**: Identity (L0, in config), Critical (L1), Standard (L2, default),
  Archive (L3). Promotion/demotion thresholds are wired in `constants.py`; the background job that
  moves rows between L1/L2/L3 is still pending (see *Roadmap*).
- **Privacy-first scopes**: `private` (only the owning agent sees it) vs `common` (all agents share
  it). Wings carry a default scope so household/personal data lands in the right place.
- **Local-first**: All data stays on your HA instance. SQLite + WAL, embeddings cached on disk.
- **Three embedding providers, graceful fallback**: a native Ollama API, an
  OpenAI-compatible endpoint (`/v1/embeddings` — llama.cpp `llama-server`, LM Studio, vLLM,
  LocalAI, Infinity, HuggingFace TEI, …), or a zero-dependency TF-IDF fallback.

## 📋 Requirements

- Home Assistant ≥ 2025.11.3 (tested against 2026.4.2)
- Python 3.14
- *(Optional, recommended)* An embedding service — either [Ollama](https://ollama.com) or any
  OpenAI-compatible server. The default model is `bge-m3` (1024-dim). `all-minilm` (384-dim)
  also works well and uses less RAM.
- Without a remote service, the integration falls back to a built-in TF-IDF engine.

## 🚀 Installation

### HACS (recommended)

Add this repository as a [custom repository](https://www.hacs.xyz/docs/faq/custom_repositories/)
in HACS, then install **AI Long Term Memory**.

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=Riscue&repository=ha-ai-memory)

### Manual

1. Copy the `custom_components/ai_memory/` directory into your HA `config/custom_components/`.
2. Restart Home Assistant.
3. Add the integration via **Settings → Devices & Services → Add Integration**.

[![Add Integration](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=ai_memory)

## ⚙️ Configuration

The config flow has two entry points: the initial setup (`AiMemoryConfigFlow`) and reconfigure via
**Configure** on the integration entry (`AiMemoryOptionsFlow`). It is a single-instance integration.

### Initial setup steps

1. **User**
   - `max_entries` — capacity before oldest entries get evicted (default `1000`).
   - `embedding_engine` — `ollama`, `openai_compatible`, or `tfidf` (no dependencies).
     Entries created before the provider split stored `remote`; on first start after the
     upgrade they are migrated automatically to `ollama` (config entry version 2 → 3).
2. **Remote config** *(only for remote providers)*
   - `remote_url` — base URL of the embedding service (default `http://127.0.0.1:11434`).
     For OpenAI-compatible servers give the base URL; the integration appends `/v1/embeddings`
     itself (e.g. `http://llama-server:8080`).
   - `api_key` — optional, sent as `Authorization: Bearer …` on every request (vLLM, TEI and
     proxied/remote endpoints).
3. **Model selection** *(only for remote providers)*
   - Ollama: lists models from `/api/tags`, then pulls the selected one via `/api/pull`
     (300 s timeout).
   - OpenAI-compatible: lists models from `/v1/models`; no pull — these servers load the model
     at startup.
   - If the server cannot be reached at all, the flow returns to the connection step with
     a `cannot_connect` error until the URL works.
   - If the server is reachable but its model list cannot be fetched, the field becomes
     free text so you can type the model name manually.
   - Default model: `bge-m3`.

Reconfiguration re-uses the same step structure, so you can switch providers, change the model,
or update the API key without removing the integration. Switching to TF-IDF clears the stored
remote settings.

> **L0 identity**: the earlier `identity_text` step is gone — its only consumer
> (`LayerManager`) is not wired into the LLM API yet. It will return via the options flow in a
> future release (see *Roadmap*).

### Storage locations

| What              | Where                                                        |
|-------------------|--------------------------------------------------------------|
| Memory DB         | `<ha_config>/ai_memory.db` (SQLite, WAL mode)                |
| TF-IDF vocabulary | `<ha_config>/.storage/ai_memory_tfidf_vocab.json` *(only if engine = tfidf)* |

> The integration owns this schema. Don't edit the DB by hand while HA is running — `WAL` journaling
> and a `busy_timeout=5000` make concurrent reads safe, but a manual write while the manager holds
> the connection can still corrupt state.

## 🤖 Usage — LLM tools (automatic)

Once installed, three tools are exposed to every Assist agent. The agent decides when to call them
based on the system prompt bundled with the integration (`llm_api/prompts.py`):

| Tool            | Purpose                                                                                  |
|-----------------|------------------------------------------------------------------------------------------|
| `add_memory`    | Proactively store durable facts. Content language follows the conversation.              |
| `search_memory` | Semantic recall before the agent answers — keeps answers consistent with stored context. |
| `delete_memory` | Remove outdated or wrong memories by room/wing/scope.                                   |

### Tool parameters (high level)

```yaml
add_memory:
  content:   "*"        # required, conversation language
  scope:     "private"  # required: private | common
  summary:   ""         # optional, 3–5 comma-separated keywords
  wing:      ""         # optional, English lowercase (auto-detected if blank)
  room:      ""         # optional, English lowercase (auto-detected if blank)

search_memory:
  query:    "*"         # required
  wing:     ""          # optional filter
  room:     ""          # optional filter
  limit:    5           # optional, 1–20

delete_memory:
  room:     ""          # at least one of room / wing / scope
  wing:     ""
  scope:    ""          # private | common
```

### Example interaction

> **User:** "I'm allergic to peanuts."
> **Agent:** *calls `add_memory(content="User is allergic to peanuts", scope="private",
> wing="personal", room="health")`*
> **Agent:** "Got it — I'll remember the peanut allergy."

A later conversation, days later:

> **User:** "What should I avoid when cooking for me?"
> **Agent:** *calls `search_memory(query="allergies food restrictions", wing="personal")` →
> hits the row stored above*
> **Agent:** "Avoid peanuts — you've told me about a peanut allergy."

## 🛠️ Usage — services (for automations/scripts)

All four services support `supports_response=OPTIONAL`, so you can capture the result into a script
response variable.

### `ai_memory.add_memory`

```yaml
service: ai_memory.add_memory
data:
  text: "The garage door code is 1234"
  wing: personal
  room: secrets
```

> The service path always saves as `scope=common`. Use the LLM `add_memory` tool (or call the
> manager directly) if you need `private` scope from an automation.

### `ai_memory.list_memories`

```yaml
service: ai_memory.list_memories
data:
  limit: 50
  wing: household
  scope: common
response_variable: memories
```

Supported filters: `limit`, `wing`, `room`, `scope`, `agent_id`.

### `ai_memory.search_memory`

```yaml
service: ai_memory.search_memory
data:
  query: "lights that should stay off after midnight"
  limit: 5
  min_score: 0.55
  wing: automation
response_variable: results
```

Supported filters: `query` (required), `limit`, `min_score`, `wing`, `room`, `agent_id`. If no
semantic match is found, the engine falls back to tokenized `LIKE` search and tags the result with
`match_type: "text"`, `score: 0.0`.

### `ai_memory.delete_memory`

```yaml
service: ai_memory.delete_memory
data:
  wing: household
  room: events
  scope: common
```

Deletes by filter (at least one of `room` / `wing` / `scope`). `private` rows can only be deleted
when the caller's `agent_id` matches the owner.

## 🏛️ Memory Palace — Wing → Room

Memories are organized into **Wings** (broad categories with a default scope) and **Rooms** (specific
topics inside a wing). Wing and room names are **always English lowercase** — this is a hard contract
with the LLM, and unknown values are auto-created rather than rejected.

Default wings seeded on first run:

| Wing         | Default scope | Default rooms                          |
|--------------|---------------|----------------------------------------|
| `household`  | `common`      | `devices`, `maintenance`, `events`     |
| `personal`   | `private`     | `preferences`, `health`, `secrets`     |
| `automation` | `common`      | `routines`, `schedules`                |
| `general`    | `common`      | `general`                              |

The agent is encouraged to pick a fitting wing/room but can leave them blank — the integration
auto-detects a room from content keywords (e.g. *"light"*, *"switch"* → `devices`) and falls back
to `general/general`.

## 🧱 Layered memory (L0–L3)

| Layer | Name      | Where it lives                                                            |
|-------|-----------|---------------------------------------------------------------------------|
| L0    | Identity  | identity text (collection paused — returns with the L0 wiring, see Roadmap) |
| L1    | Critical  | `layer=1` rows, intended as the agent's standing context (wiring pending) |
| L2    | Standard  | `layer=2` (default). The bulk of stored memories.                         |
| L3    | Archive   | `layer=3` rows. Cold storage for rarely accessed items.                   |

Promotion (L2 → L1) and demotion (L1 → L3) thresholds live in `constants.py`
(`L1_PROMOTION_THRESHOLD=10`, `L1_DEMOTION_DAYS=90`). `LayerManager.async_get_context` reads the
identity text and L1 rows, but the call site that injects this into the LLM `APIInstance` is not
wired today — both pieces exist, they're not yet connected. Until then, every new memory is written
as `layer=2` and stays there.

## 🔎 How search works

`MemorySearch.async_search` does, in order:

1. **SQL pre-filter** — `scope='common' OR (scope='private' AND agent_id=?)`, plus any `wing` /
   `room` filters.
2. **Semantic similarity** — embeds the query, computes cosine similarity with NumPy against
   stored vectors, keeps results above `min_score` (default `0.55`).
3. **Text fallback** — if nothing clears the threshold, a tokenized `LIKE` search on
   `content`+`summary` returns the best matches with `match_type="text"`, `score=0.0`.
4. **Access tracking** — every returned row gets its `access_count` and `accessed_at` bumped, which
   feeds future promotion decisions.

> **Embedding dimension is auto-detected.** Different models produce different dims (`bge-m3=1024`,
> `all-minilm=384`, TF-IDF=384). The first successful embedding persists the dimension in the
> `_meta` table; subsequent reads use the cached value. Don't hardcode `384`.

## 📡 Events & sensor

Every successful add/delete fires an `ai_memory_updated` event on the HA bus. The included
`sensor.ai_memory` (entity id derived from the name "AI Memory"; unique_id `ai_memory_store`)
listens to that event and refreshes its attributes:

- `state`: the literal string `"Active"`
- `embedding_engine`, `max_entries`, `last_updated`
- `memory_counts`: `{common, private, total}`
- `layer_distribution`: `{L0, L1, L2, L3}` counts
- `wing_distribution`: count per wing
- `palace_structure`: `{wings, rooms}` totals from the palace
- All config-entry data (provider, model name, remote URL, …)

Use it in templates, dashboards, or to trigger automations when memory changes.

## 🐛 Troubleshooting

- **Setup fails with `"Remote embedding service is not reachable"`** — the manager probes
  `/api/version` (Ollama) or `/v1/models` (OpenAI-compatible) during startup. Make sure the server
  is running at the URL you configured, or switch the provider to `tfidf` (no probe, always works).
- **`"No embedding engine available. Please check logs."`** — both the requested engine and the
  TF-IDF fallback failed to initialize. Check the HA log for the underlying cause; usually a
  missing remote URL or a permissions issue writing the TF-IDF vocab file.
- **Model not pulled** — re-enter the config flow's *Model selection* step; the integration pulls
  models on demand via `/api/pull` with a 300 s timeout (Ollama only). First pull of a large model
  can take a few minutes. OpenAI-compatible servers load the model at startup — nothing to pull.
- **Private memories not visible** — by design. `private` rows are only returned to the agent that
  owns them (filtered by `agent_id` in the SQL pre-filter). Use `scope=common` for shared facts.
- **Database corruption** — stop HA, then delete `<ha_config>/ai_memory.db` (and its `-wal` /
  `-shm` siblings). You will lose all stored memories; the schema is recreated on next start.
- **Wing/room shows up as something unexpected** — wing/room is auto-created when an unknown value
  reaches the store. If the LLM persistently picks bad names, adjust `prompts.py` (the system
  prompt) rather than trying to validate at write time.

## 🗺️ Roadmap

Things that exist in code but aren't fully wired yet:

- **L0/L1 context injection** — `LayerManager.async_get_context` will read the identity
  text (to be collected again via the options flow) and L1
  rows, but the LLM `APIInstance` doesn't call it yet. Today the agent only sees memory through the
  three tools, not as standing context.
- **Layer promotion/demotion** — thresholds (`L1_PROMOTION_THRESHOLD`, `L1_DEMOTION_DAYS`) are
  defined; the background job that promotes high-traffic L2 rows to L1 and demotes stale L1 rows to
  L3 isn't implemented. New memories always land at `layer=2`.
- **Hall/Tunnel connections** — `HallTunnelManager` exists for manual cross-room linking but isn't
  exposed through any UI/service; automatic discovery is deferred.

## 🧩 Architecture overview

```
custom_components/ai_memory/
├── __init__.py            # setup entry + service registration
├── config_flow.py         # AiMemoryConfigFlow / AiMemoryOptionsFlow
├── constants.py           # DOMAIN, DB_VERSION, engine names, defaults
├── manifest.json
├── services.yaml          # field selectors + descriptions for the 4 services
├── sensor.py              # listens to ai_memory_updated, exposes counts
├── embedding/             # EmbeddingEngine + remote/tfidf backends
├── llm_api/               # api.py (registration), tools.py, prompts.py
├── memory/                # manager, store, search, migration, layers
└── palace/                # structure, defaults, metadata, hall_tunnel (stub)
```

Two paths into the same `MemoryManager`:

1. **LLM tools** — registered via `homeassistant.helpers.llm.async_register_api`. The agent's
   `platform` is captured as `agent_id`.
2. **HA services** — same operations, exposed for automations and scripts.

Adding a new capability means updating both paths in sync: the voluptuous schema in `__init__.py`,
the matching selector in `services.yaml`, the LLM tool's `parameters` schema, and the system prompt
in `prompts.py`.

## License

MIT © [Riscue](https://github.com/Riscue)
