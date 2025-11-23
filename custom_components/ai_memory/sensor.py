import json
import logging
import os
from datetime import datetime

from homeassistant.components.sensor import SensorEntity

_LOGGER = logging.getLogger(__name__)

DOMAIN = "ai_memory"
FILE_PATH = "/tmp/ai_memory.json"


async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):
    """Set up the AI Memory sensor platform."""
    _LOGGER.info("AI Memory Sensörü kuruluyor...")
    sensor = AIMemorySensor(hass)

    # Sensör örneğini global erişime açıyoruz (__init__.py buradan erişecek)
    hass.data[DOMAIN]["sensor_instance"] = sensor

    async_add_entities([sensor], True)


class AIMemorySensor(SensorEntity):
    """Representation of the AI Memory."""

    def __init__(self, hass):
        self.hass = hass
        self._memories = []
        self._attr_name = "AI Long Term Memory"
        self._attr_unique_id = "ai_long_term_memory_01"
        self._attr_icon = "mdi:brain"

        # Dosya yoksa boş liste ile oluştur
        if not os.path.exists(FILE_PATH):
            self._save_to_file([])

    @property
    def state(self):
        """State: Anı sayısı."""
        return len(self._memories)

    @property
    def extra_state_attributes(self):
        """Attribute: Tüm metin."""
        full_text = "\n".join([f"- {m['date']}: {m['text']}" for m in self._memories])
        return {
            "full_text": full_text,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    async def async_update(self):
        """Fetch new state data."""
        # Executor ile dosya okuma (Blocking I/O engellemek için)
        self._memories = await self.hass.async_add_executor_job(self._read_file)

    def _read_file(self):
        try:
            if os.path.exists(FILE_PATH):
                with open(FILE_PATH, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    return content if isinstance(content, list) else []
        except Exception as e:
            _LOGGER.error(f"Dosya okuma hatası: {e}")
        return []

    def _save_to_file(self, data):
        try:
            with open(FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            _LOGGER.error(f"Dosya yazma hatası: {e}")

    async def async_add_memory(self, text):
        """Servis tarafından çağrılan fonksiyon."""
        entry = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "text": text
        }
        self._memories.append(entry)
        await self.hass.async_add_executor_job(self._save_to_file, self._memories)
        self.async_write_ha_state()
        _LOGGER.info(f"Hafızaya eklendi: {text}")

    async def async_clear_memory(self):
        """Hafızayı temizle."""
        self._memories = []
        await self.hass.async_add_executor_job(self._save_to_file, [])
        self.async_write_ha_state()
