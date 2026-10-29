from dataclasses import dataclass
import math

STACK_BELOW = 1000
TWO_ROW_TOOLBAR = 1500
MIN_VISIBLE = 120


@dataclass(frozen=True)
class ScreenInfo:
    name: str
    x: int
    y: int
    width: int
    height: int
    dpi: float = 96.
    ratio: float = 1.
    width_mm: float = 0.
    height_mm: float = 0.

    @property
    def area(self):
        return self.width * self.height

    @property
    def diagonal_inches(self):
        if self.width_mm <= 0 or self.height_mm <= 0:
            return None
        return math.hypot(self.width_mm, self.height_mm) / 25.4

    def contains(self, x, y):
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height


@dataclass(frozen=True)
class DisplayProfile:
    kind: str
    screens: int
    primary: ScreenInfo
    secondary: ScreenInfo | None
    font_px: int
    side_fraction: float

    @property
    def label(self):
        names = {'compact': 'tela compacta', 'laptop': 'notebook', 'desktop': 'monitor', 'wide': 'monitor amplo'}
        base = names.get(self.kind, self.kind)
        return f'{base} + {self.screens - 1} monitor(es) extra' if self.screens > 1 else f'{base} (1 monitor)'


def classify(screens, primary_index=0):
    if not screens:
        screen = ScreenInfo('padrão', 0, 0, 1366, 768)
        screens = [screen]
        primary_index = 0
    primary_index = primary_index if 0 <= primary_index < len(screens) else 0
    primary = screens[primary_index]
    diagonal = primary.diagonal_inches
    laptop = (diagonal is not None and diagonal < 16.5) or (primary.ratio >= 1.5 and primary.width <= 1600)
    if primary.width < 1100 or primary.height < 700:
        kind, font, side = 'compact', 11, .0
    elif laptop:
        kind, font, side = 'laptop', 12, .32
    elif primary.width >= 2400:
        kind, font, side = 'wide', 13, .24
    else:
        kind, font, side = 'desktop', 12, .28
    others = [s for i, s in enumerate(screens) if i != primary_index]
    secondary = max(others, key=lambda s: (s.area, -s.x, -s.y)) if others else None
    return DisplayProfile(kind, len(screens), primary, secondary, font, side)


def arrangement(width):
    width = max(0, int(width))
    return {'stacked': width < STACK_BELOW, 'toolbar_rows': 2 if width < TWO_ROW_TOOLBAR else 1}


def signature(screens):
    return ';'.join(sorted(f'{s.width}x{s.height}@{s.x},{s.y}' for s in screens)) or 'none'


def initial_geometry(profile):
    screen = profile.primary
    scale = 1. if profile.kind == 'compact' else .9
    width = max(640, int(screen.width * scale))
    height = max(480, int(screen.height * scale))
    return screen.x + (screen.width - width) // 2, screen.y + (screen.height - height) // 2, width, height


def safe_geometry(saved, screens):
    try:
        x, y, width, height = (int(v) for v in saved)
    except (TypeError, ValueError):
        return None
    if width < 320 or height < 240:
        return None
    for screen in screens:
        left, top = max(x, screen.x), max(y, screen.y)
        right, bottom = min(x + width, screen.x + screen.width), min(y + height, screen.y + screen.height)
        if right - left >= MIN_VISIBLE and bottom - top >= 40 and screen.contains(x + 40, y + 10):
            width, height = min(width, screen.width), min(height, screen.height)
            x = min(max(x, screen.x), screen.x + screen.width - width)
            y = min(max(y, screen.y), screen.y + screen.height - height)
            return x, y, width, height
    return None


def panel_geometry(screen):
    width = min(560, max(360, screen.width // 3))
    return screen.x + screen.width - width, screen.y, width, screen.height


def screens_from_qt(application):
    result = []
    for screen in application.screens():
        rect = screen.availableGeometry()
        size = screen.physicalSize()
        result.append(ScreenInfo(screen.name() or 'tela', rect.x(), rect.y(), rect.width(), rect.height(), float(screen.logicalDotsPerInch()), float(screen.devicePixelRatio()), float(size.width()), float(size.height())))
    primary = application.primaryScreen()
    index = application.screens().index(primary) if primary in application.screens() else 0
    return result, index
