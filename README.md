# 🧠 AI Long Term Memory for Home Assistant

[![Home Assistant](https://img.shields.io/badge/home%20assistant-%2341BDF5.svg?style=for-the-badge&logo=home-assistant&logoColor=white)](https://home-assistant.io)
[![hacs](https://img.shields.io/badge/HACS-Default-orange.svg?style=for-the-badge)](https://github.com/hacs/default)
[![License][license-shield]](LICENSE.md)

[license-shield]: https://img.shields.io/github/license/Riscue/ha-ai-memory.svg?style=for-the-badge

[![Active installations](https://img.shields.io/badge/dynamic/json?style=for-the-badge&color=41BDF5&logo=home-assistant&label=active%20installations&cacheSeconds=15600&url=https://analytics.home-assistant.io/custom_integrations.json&query=$.ai_memory.total)](https://github.com/Riscue/ha-ai-memory)
[![GitHub Downloads (all assets, latest release)](https://img.shields.io/github/downloads/Riscue/ha-ai-memory/latest/total?label=downloads&style=for-the-badge)](https://github.com/Riscue/ha-ai-memory/releases)

[![GitHub Release](https://img.shields.io/github/release/Riscue/ha-ai-memory.svg?style=for-the-badge)](https://github.com/Riscue/ha-ai-memory/releases)
[![GitHub Activity](https://img.shields.io/github/commit-activity/y/Riscue/ha-ai-memory.svg?style=for-the-badge)](https://github.com/Riscue/ha-ai-memory/commits/master)

![Icon](assets/logo.png)

[![Add Integration](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=ai_memory)

Long-term memory system for AI assistants in Home Assistant. Store facts, preferences, and context that persists across
conversations.

## ✨ Features

- **Native LLM Integration**: Automatically exposes tools to Home Assistant's Assist agents.
- **Multi-Engine Support**: Choose between `SentenceTransformer` (Best Quality), `FastEmbed` (RPi4 Optimized), or
  `TF-IDF` (Lightweight).
- **Scoped Memory**: Supports `private` (agent-specific) and `common` (shared) memories.
- **Privacy First**: All data is stored locally in your Home Assistant instance.
- **Offline Capable**: Works entirely offline with TF-IDF or cached models.

## 🚀 Installation

### HACS Installation (Recommended)

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=Riscue&repository=ha-ai-memory)

### Manual Installation

1. Copy `custom_components/ai_memory` to your `config/custom_components/` directory.
2. Restart Home Assistant.
3. Add the integration via Settings > Devices & Services.

[![Add Integration](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=ai_memory)

## ⚙️ Configuration

During setup (or via "Configure" on the integration entry), you can customize:

- **Storage Location**: Fixed at `/config/ai_memory/`.
- **Embedding Engine**:
    - **SentenceTransformer**: Best accuracy. Requires ~500MB RAM. Ideal for PC/NUC.
    - **FastEmbed**: Good accuracy, optimized for ARM/RPi4. Requires ~100MB RAM.
    - **TF-IDF**: Zero dependencies, very fast, lower semantic accuracy. Best for low-power hardware.
    - **Auto**: Tries engines in order: SentenceTransformer → FastEmbed → TF-IDF.

## 🤖 Usage

### For AI Agents (LLM Tools)

Once installed, the following tools are automatically available to your Assist agents:

- **`add_memory`**: Proactively saves information.
    - `content`: The text to save.
    - `scope`: `private` (default, specific to the agent) or `common` (shared household facts).
- **`search_memory`**: Retrieves relevant memories based on semantic similarity.

**Example Interaction:**
> **User:** "I'm allergic to peanuts."
> **AI:** *Calls `add_memory(content="User is allergic to peanuts", scope="private")`*
> **AI:** "I've made a note of your peanut allergy."

### For Automations (Services)

You can manage memories programmatically using Home Assistant services.

#### `ai_memory.add_memory`

Add a memory entry manually.

```yaml
service: ai_memory.add_memory
data:
  memory_id: sensor.ai_memory_store
  text: "The garage door code is 1234"
```

#### `ai_memory.list_memories`

Retrieve all memories.

```yaml
service: ai_memory.list_memories
response_variable: memories
```

#### `ai_memory.clear_memory`

Wipe all memories for a sensor.

```yaml
service: ai_memory.clear_memory
data:
  memory_id: sensor.ai_memory_store
```

## 🐛 Troubleshooting

- **"No embedding engine available"**: Ensure you have selected a supported engine for your hardware. Try switching to
  `TF-IDF` or `Auto`.
- **Import Errors**: Check logs. If using `SentenceTransformer` or `FastEmbed`, ensure dependencies are installed or
  switch to `TF-IDF`.
- **Database Issues**: If the database becomes corrupted, stop HA and delete the `ai_memory.db` file in your storage
  directory.

## License

MIT © [Riscue](https://github.com/riscue)
