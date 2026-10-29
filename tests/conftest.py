import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from hypothesis import HealthCheck, settings

settings.register_profile('default', max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.register_profile('ci', max_examples=250, deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.load_profile(os.environ.get('HYPOTHESIS_PROFILE', 'default'))


@pytest.fixture(scope='session')
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    from PySide6.QtCore import QSettings
    return QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)


@pytest.fixture
def make_window(qapp, settings):
    from app import Window
    from secure_store import VolatileStore
    created = []

    def factory(memory=None, **kwargs):
        kwargs.setdefault('settings', settings)
        window = Window(memory if memory is not None else VolatileStore(), **kwargs)
        created.append(window)
        return window

    yield factory
    for window in created:
        window.close()
