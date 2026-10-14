import pytest

from app.domain.admin_models import ADMIN_STATE, AUDIT_EVENTS


@pytest.fixture(autouse=True)
def reset_in_memory_admin_state():
    """Keep mutable admin state from leaking between otherwise independent tests."""
    ADMIN_STATE.clear()
    AUDIT_EVENTS.clear()
    yield
    ADMIN_STATE.clear()
    AUDIT_EVENTS.clear()
