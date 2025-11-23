import pytest


@pytest.mark.asyncio
async def test_integration_loads(hass):
    assert hass is not None
