"""AI Long Term Memory component."""
import logging

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.discovery import load_platform

_LOGGER = logging.getLogger(__name__)

DOMAIN = "ai_memory"
CONF_MEMORY_LOCATION = "memory_location"

CONFIG_SCHEMA = vol.Schema({
    DOMAIN: vol.Schema(
        {
            vol.Optional(CONF_MEMORY_LOCATION): cv.string
        }
    )
}, extra=vol.ALLOW_EXTRA)


async def async_setup(hass: HomeAssistant, config: dict):
    """Set up the AI Memory component."""
    _LOGGER.info("AI Memory bileşeni başlatılıyor...")

    # Global veri deposunu oluştur
    hass.data.setdefault(DOMAIN, {})

    # 1. Sensör platformunu manuel tetikle
    # Bu, sensor.py dosyasındaki async_setup_platform'u çalıştırır
    load_platform(hass, "sensor", DOMAIN, {}, config)

    # 2. Servisleri Tanımla (Handler fonksiyonları)
    async def handle_add_memory(call):
        text = call.data.get("text")
        sensor_instance = hass.data[DOMAIN].get("sensor_instance")

        if sensor_instance:
            await sensor_instance.async_add_memory(text)
        else:
            _LOGGER.error("AI Memory sensörü henüz hazır değil, kayıt yapılamadı!")

    async def handle_clear_memory(call):
        sensor_instance = hass.data[DOMAIN].get("sensor_instance")
        if sensor_instance:
            await sensor_instance.async_clear_memory()

    # 3. Servisleri Kaydet
    hass.services.async_register(DOMAIN, "add_memory", handle_add_memory)
    hass.services.async_register(DOMAIN, "clear_memory", handle_clear_memory)

    _LOGGER.info("AI Memory servisleri kaydedildi.")
    return True
