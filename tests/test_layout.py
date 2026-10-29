import pytest
from hypothesis import given, settings as hsettings, strategies as st
from PySide6.QtCore import Qt
from layout import ScreenInfo, arrangement, classify, initial_geometry, panel_geometry, safe_geometry, signature

LAPTOP = ScreenInfo('eDP-1', 0, 0, 1536, 824, 96., 1.25, 309., 174.)
RETINA = ScreenInfo('Built-in', 0, 25, 1440, 875, 72., 2., 0., 0.)
DESKTOP = ScreenInfo('DP-1', 0, 0, 1920, 1040, 96., 1., 527., 296.)
WIDE = ScreenInfo('DP-2', 1920, 0, 3440, 1400, 96., 1., 800., 335.)
SMALL = ScreenInfo('HDMI', 0, 0, 1024, 600)


@pytest.mark.parametrize('screen,kind', [(LAPTOP, 'laptop'), (RETINA, 'laptop'), (DESKTOP, 'desktop'), (WIDE, 'wide'), (SMALL, 'compact')])
def test_classification(screen, kind):
    profile = classify([screen])
    assert profile.kind == kind and profile.secondary is None and profile.screens == 1


def test_multi_monitor_picks_largest_secondary():
    profile = classify([LAPTOP, DESKTOP, WIDE], 0)
    assert profile.kind == 'laptop' and profile.secondary == WIDE and profile.screens == 3
    assert 'monitor' in profile.label
    assert classify([], 5).primary.width == 1366


def test_breakpoints():
    assert arrangement(900) == {'stacked': True, 'toolbar_rows': 2}
    assert arrangement(1200) == {'stacked': False, 'toolbar_rows': 2}
    assert arrangement(1800) == {'stacked': False, 'toolbar_rows': 1}


def test_signature_is_order_independent():
    assert signature([LAPTOP, WIDE]) == signature([WIDE, LAPTOP]) != signature([LAPTOP])


def test_geometry_lost_with_unplugged_monitor_is_not_restored():
    saved = (2200, 100, 1600, 900)
    assert safe_geometry(saved, [DESKTOP, WIDE]) == saved
    assert safe_geometry(saved, [DESKTOP]) is None
    assert safe_geometry((100, -500, 800, 600), [DESKTOP]) is None
    assert safe_geometry((1800, 100, 1000, 900), [DESKTOP]) == (920, 100, 1000, 900)
    assert safe_geometry(('x', 1, 2, 3), [DESKTOP]) is None
    assert safe_geometry((0, 0, 10, 10), [DESKTOP]) is None


@given(st.tuples(st.integers(-10000, 10000), st.integers(-10000, 10000), st.integers(-100, 8000), st.integers(-100, 8000)))
@hsettings(max_examples=300, deadline=None)
def test_restored_geometry_always_fits_a_screen(rect):
    result = safe_geometry(rect, [LAPTOP, WIDE])
    if result is not None:
        x, y, w, h = result
        assert any(s.x <= x and s.y <= y and x + w <= s.x + s.width and y + h <= s.y + s.height for s in (LAPTOP, WIDE))


def test_initial_and_panel_geometry_inside_screen():
    for screen in (LAPTOP, DESKTOP, WIDE, SMALL):
        for x, y, w, h in (initial_geometry(classify([screen])), panel_geometry(screen)):
            assert screen.x <= x and screen.y <= y and x + w <= screen.x + screen.width and y + h <= screen.y + screen.height


def test_window_reflows_with_width(qapp, make_window):
    window = make_window()
    window.show()
    window.setWindowState(Qt.WindowState.WindowNoState)
    window.resize(780, 700)
    qapp.processEvents()
    assert window.split.orientation() == Qt.Orientation.Vertical
    assert window.tabbed and window.tabs.count() == 3
    window.resize(1700, 900)
    qapp.processEvents()
    assert window.split.orientation() == Qt.Orientation.Horizontal
    assert not window.tabbed and window.metrics.parentWidget() is window.side
    window.close()


def test_window_geometry_is_remembered(qapp, make_window, settings):
    window = make_window()
    window.show()
    window.setWindowState(Qt.WindowState.WindowNoState)
    window.setGeometry(40, 60, 700, 600)
    qapp.processEvents()
    window.close()
    assert [int(v) for v in settings.value(window._geometry_key())] == [40, 60, 700, 600]
    again = make_window()
    again.setWindowState(Qt.WindowState.WindowNoState)
    again.profile = classify([DESKTOP])
    again.restore_geometry([DESKTOP])
    rect = again.geometry()
    assert (rect.x(), rect.y(), rect.width(), rect.height()) == (40, 60, 700, 600)


def test_toolbar_wraps_instead_of_forcing_width(qapp, make_window):
    window = make_window()
    window.show()
    window.setWindowState(Qt.WindowState.WindowNoState)
    window.resize(700, 600)
    qapp.processEvents()
    assert window.minimumSizeHint().width() <= 700
    assert window.toolbar.heightForWidth(600) > window.toolbar.heightForWidth(1900)


def test_side_panel_detaches_to_secondary_screen_and_returns(qapp, make_window):
    window = make_window()
    window.show()
    window.profile = classify([DESKTOP, WIDE])
    window.panel_button.setVisible(True)
    window.panel_button.setChecked(True)
    assert window.panel_window is not None and window.side.window() is window.panel_window
    assert window.panel_window.geometry().x() >= WIDE.x
    window.panel_window.close()
    qapp.processEvents()
    assert window.panel_window is None and window.side.window() is window
    assert window.split.indexOf(window.side) == 1
    window.panel_button.setChecked(True)
    window.close()
    assert window.panel_window is None
