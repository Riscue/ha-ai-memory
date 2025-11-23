# 🧠 AI Long Term Memory

[![Home Assistant](https://img.shields.io/badge/home%20assistant-%2341BDF5.svg?style=for-the-badge&logo=home-assistant&logoColor=white)](https://home-assistant.io)
[![hacs](https://img.shields.io/badge/HACS-Default-orange.svg?style=for-the-badge)](https://github.com/hacs/default)
[![License][license-shield]](LICENSE.md)

[license-shield]: https://img.shields.io/github/license/Riscue/ha-ai-memory.svg?style=for-the-badge

[![Active installations](https://img.shields.io/badge/dynamic/json?style=for-the-badge&color=41BDF5&logo=home-assistant&label=active%20installations&cacheSeconds=15600&url=https://analytics.home-assistant.io/custom_integrations.json&query=$.ai_memory.total)](https://github.com/Riscue/ha-ai-memory)
[![GitHub Downloads (all assets, latest release)](https://img.shields.io/github/downloads/Riscue/ha-ai-memory/latest/total?label=downloads&style=for-the-badge)](https://github.com/Riscue/ha-ai-memory/releases)

[![GitHub Release](https://img.shields.io/github/release/Riscue/ha-ai-memory.svg?style=for-the-badge)](https://github.com/Riscue/ha-ai-memory/releases)
[![GitHub Activity](https://img.shields.io/github/commit-activity/y/Riscue/ha-ai-memory.svg?style=for-the-badge)](https://github.com/Riscue/ha-ai-memory/commits/master)

Long-term memory system for AI assistants in Home Assistant. Store facts, preferences, and context that persists across
conversations.

## ✨ Features

- 📝 **Multiple Memories**: Create separate memory banks (Personal, Work, Shopping, etc.)
- 🤖 **AI Integration**: Seamlessly integrates with Extended OpenAI Conversation
- 💾 **Persistent Storage**: JSON-based storage that survives restarts
- 🔄 **Easy Management**: Simple services to add, clear, and list memories
- 🎯 **Config Flow**: User-friendly UI configuration
- 📊 **Sensors**: Each memory is a sensor with rich attributes

## 📦 Installation

### HACS (Recommended)

1. Open HACS
2. Go to Integrations
3. Click the three dots (top right)
4. Custom repositories
5. Add this repository URL
6. Install "AI Memory"
7. Restart Home Assistant

### Manual

1. Copy `custom_components/ai_memory` folder to your `config/custom_components/`
2. Restart Home Assistant
3. Go to Settings → Integrations → Add Integration
4. Search for "AI Memory"

## 🚀 Quick Start

### 1. Add Your First Memory

```
Settings → Integrations → Add Integration → AI Memory
```

- Enter memory name (e.g., "Personal")
- Set storage location (default: `/config/ai_memory/`)
- Set max entries (default: 1000)

### 2. Add Content to Memory

```yaml
service: ai_memory.add_memory
data:
  memory_id: personal
  text: "User prefers coffee at 7 AM"
```

### 3. Use with AI Assistant

Add this to your Extended OpenAI Conversation prompt:

```jinja
{% set memory = state_attr('sensor.ai_memory_personal', 'prompt_context_snippet') %}
{{ memory }}
```

## 📖 Services

### `ai_memory.add_memory`

Add a new entry to memory

```yaml
service: ai_memory.add_memory
data:
  memory_id: personal
  text: "User likes dark mode"
```

### `ai_memory.clear_memory`

Clear all entries from a memory

```yaml
service: ai_memory.clear_memory
data:
  memory_id: personal
```

### `ai_memory.list_memories`

Get all memories with details

```yaml
service: ai_memory.list_memories
response_variable: memories
```

## 🎯 Use Cases

### Personal Assistant

```yaml
# Remember preferences
service: ai_memory.add_memory
data:
  memory_id: personal
  text: "Prefers 22°C temperature, dislikes loud music"
```

### Work Tasks

```yaml
# Track deadlines
service: ai_memory.add_memory
data:
  memory_id: work
  text: "Q1 presentation due February 15th"
```

### Shopping List

```yaml
# Maintain shopping needs
service: ai_memory.add_memory
data:
  memory_id: shopping
  text: "Need milk, eggs, and bread"
```

## 🤖 AI Integration

### Extended OpenAI Conversation

Add to your system prompt:

```jinja
{% set personal_memory = state_attr('sensor.ai_memory_personal', 'prompt_context_snippet') %}
{% if personal_memory %}
{{ personal_memory }}
{% endif %}

You are a helpful assistant with access to long-term memory.
Use the memories above to provide personalized responses.
```

### Example Conversation

```
User: "What's my preferred temperature?"
AI: "Based on your preferences, you like 22°C."

User: "Remember that I prefer tea over coffee now"
[Automation adds to memory]

User: "What do I prefer to drink?"
AI: "You prefer tea over coffee."
```

## 📊 Sensors

Each memory creates a sensor:

- **Entity ID**: `sensor.ai_memory_{memory_id}`
- **State**: Number of entries
- **Attributes**:
    - `full_text`: All memories formatted
    - `prompt_context_snippet`: Ready-to-use context for AI
    - `memory_id`: Unique identifier
    - `memory_name`: Display name
    - `entry_count`: Number of entries
    - `max_entries`: Maximum allowed entries
    - `last_updated`: Last modification time

## 🔧 Configuration

### Options (Per Memory)

- **Memory Name**: Display name
- **Max Entries**: Limit (1-10000)
    - When exceeded, oldest entries are removed

### Storage

- Default location: `/config/ai_memory/`
- File format: `{memory_id}.json`
- Encoding: UTF-8
- Format: JSON array with timestamp and text

## 🎨 Lovelace Examples

### Simple Card

```yaml
type: markdown
title: "🧠 Personal Memory"
content: |
  **Entries:** {{ states('sensor.ai_memory_personal') }}

  {{ state_attr('sensor.ai_memory_personal', 'full_text') }}
```

### Entities Card

```yaml
type: entities
title: AI Memories
entities:
  - sensor.ai_memory_personal
  - sensor.ai_memory_work
  - sensor.ai_memory_shopping
```

## 🔄 Automation Examples

### Auto-save Preferences

```yaml
automation:
  - alias: "Save Temperature Preference"
    trigger:
      - platform: state
        entity_id: climate.living_room
        attribute: temperature
    action:
      - service: ai_memory.add_memory
        data:
          memory_id: personal
          text: "Preferred temperature set to {{ state_attr('climate.living_room', 'temperature') }}°C"
```

### Voice Commands

```yaml
automation:
  - alias: "Remember via Voice"
    trigger:
      - platform: conversation
        command: "remember that *"
    action:
      - service: ai_memory.add_memory
        data:
          memory_id: personal
          text: "{{ trigger.slots.statement }}"
```

## 📁 File Structure

```
custom_components/ai_memory/
├── __init__.py                 # Main integration
├── config_flow.py             # UI configuration
├── sensor.py                  # Sensor platform
├── extended_openai_helper.py  # AI integration helper
├── conversation.py            # Conversation intents
├── manifest.json              # Integration manifest
├── services.yaml              # Service definitions
└── translations/
    └── en.json               # English translations
```

## 🐛 Troubleshooting

### Memory not saving

- Check file permissions on `/config/ai_memory/`
- Check logs: Settings → System → Logs
- Verify `memory_id` is correct

### AI not using memory

- Verify sensor state: Developer Tools → States
- Check `prompt_context_snippet` attribute
- Ensure prompt template includes memory

### Services not appearing

- Restart Home Assistant
- Check Developer Tools → Services
- Verify integration is loaded

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 🙏 Credits

Created for Home Assistant community
Built with ❤️ for better AI assistants

## 📮 Support

- Issues: [GitHub Issues](https://github.com/yourusername/ai_memory/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/ai_memory/discussions)
- Community: [Home Assistant Forum](https://community.home-assistant.io/)

## License

MIT © [Riscue](https://github.com/riscue)
