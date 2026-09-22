"""
MacroLens - AI macro tracker (Flet, single file)

Run:
    pip install "flet>=1.0" google-genai pillow     # pillow is optional (image downscaling)
    export GEMINI_API_KEY="your-key"                # optional: paste it in-app instead
    flet run main.py

Notes
- Dark-only "Slick Athletic Cyber-Minimalism" theme.
- Data lives in one zlib-compressed, compact-key JSON blob inside the app's private
  storage directory (FLET_APP_STORAGE_DATA on device, ~/.macrolens on desktop).
- Without a Gemini key the app falls back to realistic randomized estimates so the
  whole flow can be exercised offline.
- Optional env vars: GEMINI_API_KEY (or GOOGLE_API_KEY), GEMINI_MODEL.
"""
from __future__ import annotations
from builtins import str, int, float, bool, isinstance, max, min, round, Exception, ValueError, TypeError

"""
Optional env vars: GEMINI_API_KEY (or GOOGLE_API_KEY), GEMINI_MODEL.
"""

# All your other standard imports follow below here safely...
import asyncio
import base64
import io
import json
import math
import os
import random
import re
import time
import uuid
import zlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import flet as ft


# The Gemini SDK is optional at import time so the UI still boots without it.
try:
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover - depends on the environment
    genai = None
    genai_types = None


# =============================================================================
# 1. DESIGN TOKENS
# =============================================================================
APP_NAME = "MacroLensAI"

BG = "#121212"            # matte black canvas
SURFACE = "#1E1E1E"       # dark gray cards
SURFACE_LO = "#171717"    # wells / inputs inside cards
SURFACE_HI = "#282828"    # raised chips
CRIMSON = "#FF2E63"       # primary accent (calories)
AMBER = "#FF9F43"         # secondary accent (protein)
CARB_C = "#6C7BFF"        # supporting tints so four rings stay legible
FAT_C = "#2EE6A6"
WATER_C = "#35C9F5"
TEXT = "#FFFFFF"
TEXT_DIM = "#A9A9B3"
TEXT_MUTE = "#6F6F79"

EVENING_HOUR = 19         # after this hour an unmet protein goal raises a banner
LOW_CONFIDENCE = 0.75     # below this the AI result asks the user to confirm
DEFAULT_MODEL = "gemini-2.5-flash"

# Meal provenance codes (stored as small ints)
SRC_SAMPLE, SRC_AI, SRC_CORRECTED, SRC_ESTIMATE = 0, 1, 2, 3
SRC_LABEL = {SRC_SAMPLE: "Sample", SRC_AI: "AI scan", SRC_CORRECTED: "Corrected", SRC_ESTIMATE: "Estimate"}


def alpha(color: str, a: float) -> str:
    """Colour with opacity."""
    return ft.Colors.with_opacity(a, color)


# =============================================================================
# 2. SAFE NUMBER HELPERS (NaN / zero-division guards)
# =============================================================================
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def num(value, default=0.0, lo: Optional[float] = None, hi: Optional[float] = None):
    """Coerce anything to a finite float, clamp it, or return `default`."""
    try:
        if value is None or isinstance(value, bool):
            return default
        if isinstance(value, str):
            m = _NUM_RE.search(value.replace(",", ""))
            if not m:
                return default
            value = m.group(0)
        x = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x):
        return default
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x


def ratio(part: float, whole: float) -> float:
    """part/whole clamped to [0, 1]; safe when whole is zero, NaN or negative."""
    part, whole = num(part), num(whole)
    if whole <= 0:
        return 0.0
    return max(0.0, min(1.0, part / whole))


def fmt(n: float) -> str:
    return f"{int(round(num(n))):,}"


def fmt_time(dt: datetime) -> str:
    h = dt.hour % 12 or 12
    return f"{h}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


def day_label(d: date) -> str:
    today = date.today()
    if d == today:
        return "Today"
    if d == today - timedelta(days=1):
        return "Yesterday"
    return f"{d.strftime('%a')}, {d.day} {d.strftime('%b')}"


_GLYPHS = [
    (("chicken", "wing"), "🍗"), (("salad", "poke"), "🥗"), (("pizza",), "🍕"), (("burger",), "🍔"),
    (("egg", "omelet"), "🍳"), (("oat", "porridge", "parfait", "granola"), "🥣"), (("dal", "curry", "paneer"), "🍛"),
    (("rice", "bowl"), "🍚"), (("pasta", "spaghetti", "noodle", "ramen"), "🍝"), (("steak", "beef", "lamb"), "🥩"),
    (("fish", "salmon", "tuna"), "🐟"), (("wrap", "burrito", "taco", "sandwich"), "🌯"),
    (("momo", "dumpling"), "🥟"), (("smoothie", "shake", "protein"), "🥤"), (("yogurt", "milk"), "🥛"),
    (("banana", "apple", "fruit", "berry"), "🍌"), (("bread", "toast"), "🍞"), (("soup",), "🍲"),
    (("sushi",), "🍣"), (("cake", "dessert", "ice cream", "cookie"), "🍰"), (("coffee", "latte"), "☕"),
]


def glyph(name: str) -> str:
    n = (name or "").lower()
    for keys, emoji in _GLYPHS:
        if any(k in n for k in keys):
            return emoji
    return "🍽️"


# =============================================================================
# 3. DATA MODEL
# =============================================================================
@dataclass
class Targets:
    calories: int = 2400
    protein: int = 160
    carbs: int = 260
    fats: int = 70
    water: int = 3000


@dataclass
class Meal:
    id: str
    ts: int                 # unix seconds, local time is derived on display
    name: str
    calories: float
    protein: float
    carbs: float
    fats: float
    conf: float = 1.0
    src: int = SRC_AI

    @property
    def when(self) -> datetime:
        return datetime.fromtimestamp(self.ts)


@dataclass
class Analysis:
    """Result of an image scan or a text lookup, before it becomes a Meal."""
    dish: str
    confidence: float
    calories: float
    protein: float
    carbs: float
    fats: float
    origin: str = "ai"       # "ai" or "estimate"
    note: str = ""
    corrected: bool = False

    @property
    def mismatch(self) -> bool:
        """Calories that disagree with 4/4/9 macro maths by a wide margin."""
        expected = 4 * self.protein + 4 * self.carbs + 9 * self.fats
        if self.calories <= 0 or expected <= 0:
            return True
        return abs(self.calories - expected) > max(90.0, 0.35 * max(self.calories, expected))

    @property
    def needs_review(self) -> bool:
        vague = self.dish.strip().lower() in ("", "unknown", "unknown dish", "n/a", "none")
        return self.confidence < LOW_CONFIDENCE or self.mismatch or vague


# =============================================================================
# 4. LIGHTWEIGHT STORAGE (inline "storage_manager")
# =============================================================================
class StorageManager:
    """
    One compressed file, compact keys:
        t : [calories, protein, carbs, fats, water]            targets
        m : [[id, ts, name, kcal, p, c, f, conf, src], ...]    meals (newest last)
        w : {"YYYY-MM-DD": ml}                                 water per day
        k : api key typed in-app (optional)
    """

    MAGIC = b"ML1"
    FILE_NAME = "macrolens.dat"
    MAX_MEALS = 600

    def __init__(self) -> None:
        self.path = self._resolve_path()
        self.targets = Targets()
        self.meals: list[Meal] = []
        self.water: dict[str, int] = {}
        self.api_key = ""
        self.load()

    # ---- location ---------------------------------------------------------
    @staticmethod
    def _resolve_path() -> Path:
        base = os.getenv("FLET_APP_STORAGE_DATA")
        candidates = [Path(base)] if base else []
        candidates += [Path.home() / ".macrolens", Path.cwd() / ".macrolens"]
        for folder in candidates:
            try:
                folder.mkdir(parents=True, exist_ok=True)
                probe = folder / ".w"
                probe.write_text("ok")
                probe.unlink()
                return folder / StorageManager.FILE_NAME
            except Exception:
                continue
        return Path(StorageManager.FILE_NAME)

    # ---- persistence ------------------------------------------------------
    def load(self) -> None:
        try:
            blob = self.path.read_bytes()
            if not blob.startswith(self.MAGIC):
                raise ValueError("bad header")
            data = json.loads(zlib.decompress(blob[len(self.MAGIC):]).decode("utf-8"))
            self._decode(data)
        except FileNotFoundError:
            self._seed()
            self.save()
        except Exception:
            # Corrupt or incompatible file: start clean rather than crash.
            self.targets, self.meals, self.water, self.api_key = Targets(), [], {}, ""
            self._seed()
            self.save()

    def save(self) -> None:
        try:
            payload = {
                "t": [self.targets.calories, self.targets.protein, self.targets.carbs,
                      self.targets.fats, self.targets.water],
                "m": [[m.id, m.ts, m.name, round(m.calories), round(m.protein, 1), round(m.carbs, 1),
                       round(m.fats, 1), round(m.conf, 2), m.src] for m in self.meals[-self.MAX_MEALS:]],
                "w": self._pruned_water(),
                "k": self.api_key,
            }
            raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            tmp = self.path.with_suffix(".tmp")
            tmp.write_bytes(self.MAGIC + zlib.compress(raw, 9))
            os.replace(tmp, self.path)  # atomic swap so a crash never leaves half a file
        except Exception:
            pass  # storage problems must never take the UI down

    def _pruned_water(self) -> dict[str, int]:
        cutoff = (date.today() - timedelta(days=90)).isoformat()
        return {k: v for k, v in self.water.items() if k >= cutoff}

    def _decode(self, data: dict) -> None:
        t = data.get("t") or []
        if len(t) == 5:
            self.targets = Targets(
                int(num(t[0], 2400, 500, 8000)), int(num(t[1], 160, 10, 500)), int(num(t[2], 260, 10, 1000)),
                int(num(t[3], 70, 5, 400)), int(num(t[4], 3000, 250, 10000)))
        meals: list[Meal] = []
        for row in data.get("m") or []:
            try:
                meals.append(Meal(str(row[0]), int(row[1]), str(row[2]), num(row[3]), num(row[4]),
                                  num(row[5]), num(row[6]), num(row[7], 1.0, 0, 1), int(row[8])))
            except Exception:
                continue
        self.meals = sorted(meals, key=lambda m: m.ts)
        self.water = {str(k): int(num(v)) for k, v in (data.get("w") or {}).items()}
        self.api_key = str(data.get("k") or "")

    # ---- first-run demo data ----------------------------------------------
    def _seed(self) -> None:
        rng = random.Random(21)
        now = datetime.now()
        pool = [d for d in MOCK_DISHES]
        day_slots = [(7, 40), (12, 50), (16, 10), (19, 45)]
        meals: list[Meal] = []
        for back in range(4, 0, -1):
            d = (now - timedelta(days=back)).replace(second=0, microsecond=0)
            for hh, mm in day_slots:
                name, cal, p, c, f = rng.choice(pool)
                k = rng.uniform(0.92, 1.1)
                stamp = d.replace(hour=hh, minute=mm)
                meals.append(Meal(uuid.uuid4().hex[:8], int(stamp.timestamp()), name, cal * k, p * k, c * k,
                                  f * k, 0.9, SRC_SAMPLE))
            self.water[d.date().isoformat()] = rng.randrange(2000, 3200, 250)
        start = now.replace(hour=0, minute=5, second=0, microsecond=0)
        for name, cal, p, c, f, mins_ago in (
            ("Protein oats with banana", 450, 34, 58, 10, 330),
            ("Grilled chicken rice bowl", 620, 46, 62, 17, 120),
        ):
            stamp = max(start, now - timedelta(minutes=mins_ago))
            meals.append(Meal(uuid.uuid4().hex[:8], int(stamp.timestamp()), name, cal, p, c, f, 0.92, SRC_SAMPLE))
        self.water[date.today().isoformat()] = 750
        self.meals = sorted(meals, key=lambda m: m.ts)

    # ---- queries and mutations -----------------------------------------------
    def meals_on(self, d: date) -> list[Meal]:
        return [m for m in self.meals if m.when.date() == d]

    def totals_on(self, d: date) -> dict:
        ms = self.meals_on(d)
        return {
            "cal": sum(num(m.calories) for m in ms), "protein": sum(num(m.protein) for m in ms),
            "carbs": sum(num(m.carbs) for m in ms), "fats": sum(num(m.fats) for m in ms),
        }

    def water_on(self, d: date) -> int:
        return int(num(self.water.get(d.isoformat(), 0), 0, 0))

    def add_water(self, d: date, ml: int) -> None:
        self.water[d.isoformat()] = max(0, self.water_on(d) + ml)
        self.save()

    def add_meal(self, meal: Meal) -> None:
        self.meals.append(meal)
        self.meals.sort(key=lambda m: m.ts)
        self.save()

    def remove_meal(self, meal_id: str) -> Optional[Meal]:
        for m in self.meals:
            if m.id == meal_id:
                self.meals.remove(m)
                self.save()
                return m
        return None


# =============================================================================
# 5. GEMINI SERVICE (structured schema -> loose JSON -> offline estimate)
# =============================================================================
MOCK_DISHES = [
    ("Grilled chicken rice bowl", 620, 46, 62, 17),
    ("Paneer tikka wrap", 540, 28, 48, 26),
    ("Salmon avocado poke bowl", 590, 38, 52, 24),
    ("Steamed chicken momo (10 pieces)", 430, 26, 46, 14),
    ("Dal bhat with vegetables", 680, 24, 112, 14),
    ("Greek yogurt berry parfait", 280, 22, 34, 6),
    ("Beef burger and fries", 890, 38, 84, 44),
    ("Egg white veggie omelette", 310, 32, 8, 16),
    ("Margherita pizza (2 slices)", 520, 22, 64, 20),
    ("Protein oats with banana", 450, 34, 58, 10),
    ("Chicken Caesar salad", 480, 36, 18, 28),
    ("Peanut butter banana smoothie", 410, 22, 52, 14),
]

IMAGE_SYSTEM_PROMPT = (
    "You are a sports-nutrition analyst. Look at the meal photo and estimate the WHOLE portion shown. "
    "Reply with ONE JSON object and nothing else, matching exactly this shape: "
    '{"identifiedDish": "Name", "confidence": 0.95, "calories": 450, "protein": 35, "carbs": 40, "fats": 12}. '
    "identifiedDish is a specific, human-readable dish name. confidence is 0.0-1.0 and reflects how sure you are "
    "of the dish and portion. calories are kcal; protein, carbs and fats are grams. Calories should roughly equal "
    "4*protein + 4*carbs + 9*fats. If several items are on the plate, combine them and name the plate. "
    "If the image is not food, use identifiedDish 'Unknown' with confidence 0 and zeros. No markdown, no prose."
)

TEXT_SYSTEM_PROMPT = (
    "You are a sports-nutrition analyst. The user names a dish. Estimate the macros for one typical serving "
    "unless a quantity is given in the name. Reply with ONE JSON object and nothing else: "
    '{"identifiedDish": "Name", "confidence": 0.9, "calories": 450, "protein": 35, "carbs": 40, "fats": 12}. '
    "Calories are kcal, the rest are grams, and calories should roughly equal 4*protein + 4*carbs + 9*fats. "
    "No markdown, no prose."
)


def sniff_mime(raw: bytes) -> str:
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1"):
        return "image/heic"
    return "image/jpeg"


def prepare_image(raw: bytes) -> tuple[bytes, str]:
    """Downscale to <=1280px JPEG when Pillow is present; otherwise pass bytes through."""
    try:
        from PIL import Image, ImageOps

        im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
        im.thumbnail((1280, 1280))
        if im.mode != "RGB":
            im = im.convert("RGB")
        out = io.BytesIO()
        im.save(out, "JPEG", quality=82, optimize=True)
        return out.getvalue(), "image/jpeg"
    except Exception:
        return raw, sniff_mime(raw)


def parse_json_loose(text: str) -> dict:
    """Pull a JSON object out of a model reply, tolerating fences and stray prose."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    candidates = [text]
    braces = re.search(r"\{.*\}", text, re.DOTALL)
    if braces:
        candidates.append(braces.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            obj = obj[0]
        if isinstance(obj, dict):
            return obj
    raise ValueError("Model reply was not valid JSON")


def normalize_analysis(raw: dict, fallback_name: str = "", origin: str = "ai") -> Analysis:
    def pick(*keys):
        for k in keys:
            if k in raw and raw[k] is not None:
                return raw[k]
        return None

    conf = num(pick("confidence", "conf"), 0.0)
    if conf > 1.0:  # model answered 0-100
        conf = conf / 100.0
    name = str(pick("identifiedDish", "dish", "name") or fallback_name or "Unknown dish").strip()[:80]
    return Analysis(
        dish=name or "Unknown dish",
        confidence=max(0.0, min(1.0, conf)),
        calories=num(pick("calories", "kcal"), 0.0, 0, 6000),
        protein=num(pick("protein"), 0.0, 0, 400),
        carbs=num(pick("carbs", "carbohydrates"), 0.0, 0, 800),
        fats=num(pick("fats", "fat"), 0.0, 0, 400),
        origin=origin,
    )


class AIService:
    def __init__(self, store: StorageManager) -> None:
        self.store = store
        self._client = None
        self._client_key = ""
        self.last_error = ""

    # ---- configuration ----------------------------------------------------
    @property
    def env_key(self) -> str:
        return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()

    @property
    def api_key(self) -> str:
        return self.env_key or (self.store.api_key or "").strip()

    @property
    def live(self) -> bool:
        return bool(self.api_key) and genai is not None

    @property
    def model(self) -> str:
        return os.getenv("GEMINI_MODEL") or DEFAULT_MODEL

    def _get_client(self):
        key = self.api_key
        if self._client is None or key != self._client_key:
            self._client = genai.Client(api_key=key)
            self._client_key = key
        return self._client

    # ---- public API (blocking; call from a worker thread) ------------------
    def analyze_image(self, image: bytes, mime: str) -> Analysis:
        if not self.live:
            return self._mock_scan()
        try:
            part = genai_types.Part.from_bytes(data=image, mime_type=mime)
            raw = self._generate([part, "Analyze this meal and return the JSON object."], IMAGE_SYSTEM_PROMPT)
            return normalize_analysis(raw)
        except Exception as ex:
            return self._fail(ex, self._mock_scan())

    def lookup_dish(self, dish: str) -> Analysis:
        dish = dish.strip()
        if not self.live:
            return self._mock_lookup(dish)
        try:
            raw = self._generate([f"Dish: {dish}"], TEXT_SYSTEM_PROMPT)
            result = normalize_analysis(raw, fallback_name=dish)
            result.dish = dish  # the user's wording wins
            return result
        except Exception as ex:
            return self._fail(ex, self._mock_lookup(dish))

    # ---- internals ---------------------------------------------------------
    def _fail(self, ex: Exception, fallback: Analysis) -> Analysis:
        self.last_error = f"{type(ex).__name__}: {ex}"[:200]
        fallback.note = "Gemini could not be reached, so this is an estimate."
        return fallback

    @staticmethod
    def _schema():
        T, S = genai_types.Type, genai_types.Schema
        return S(
            type=T.OBJECT,
            properties={
                "identifiedDish": S(type=T.STRING, description="Specific dish name"),
                "confidence": S(type=T.NUMBER, description="0.0 to 1.0"),
                "calories": S(type=T.INTEGER, description="kcal"),
                "protein": S(type=T.NUMBER, description="grams"),
                "carbs": S(type=T.NUMBER, description="grams"),
                "fats": S(type=T.NUMBER, description="grams"),
            },
            required=["identifiedDish", "confidence", "calories", "protein", "carbs", "fats"],
        )

    def _generate(self, contents: list, system: str) -> dict:
        client = self._get_client()
        try:  # attempt 1: schema-enforced JSON
            resp = client.models.generate_content(
                model=self.model, contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system, response_mime_type="application/json",
                    response_schema=self._schema(), temperature=0.2),
            )
            return parse_json_loose(resp.text)
        except Exception:
            # attempt 2: plain prompt, tolerant parser (older models, schema hiccups)
            resp = client.models.generate_content(
                model=self.model, contents=contents,
                config=genai_types.GenerateContentConfig(system_instruction=system, temperature=0.2),
            )
            return parse_json_loose(resp.text)

    # ---- offline stand-ins -------------------------------------------------
    @staticmethod
    def _jitter(name: str, p: float, c: float, f: float, rng: random.Random, conf: float) -> Analysis:
        k = lambda: rng.uniform(0.92, 1.08)  # noqa: E731
        p, c, f = p * k(), c * k(), f * k()
        return Analysis(name, conf, round(4 * p + 4 * c + 9 * f), round(p, 1), round(c, 1), round(f, 1),
                        origin="estimate", note="Sample estimate. Add a Gemini API key for real scans.")

    def _mock_scan(self) -> Analysis:
        rng = random.Random()
        name, cal, p, c, f = rng.choice(MOCK_DISHES)
        # roughly one in four mock scans is low-confidence so the correction flow is easy to test
        conf = rng.uniform(0.45, 0.72) if rng.random() < 0.25 else rng.uniform(0.82, 0.97)
        return self._jitter(name, p, c, f, rng, round(conf, 2))

    def _mock_lookup(self, dish: str) -> Analysis:
        low = dish.lower()
        query = set(re.findall(r"[a-z]+", low))
        best, best_score = None, 0
        for name, _cal, p, c, f in MOCK_DISHES:
            score = len(query & {w for w in re.findall(r"[a-z]+", name.lower()) if len(w) > 3})
            if score > best_score:
                best, best_score = (p, c, f), score
        rng = random.Random(zlib.crc32(low.encode()))
        if best:
            return self._jitter(dish, *best, rng, 0.9)
        p, c, f = rng.uniform(12, 45), rng.uniform(15, 85), rng.uniform(6, 30)
        return self._jitter(dish, p, c, f, rng, 0.8)


# =============================================================================
# 6. SMALL UI BUILDERS
# =============================================================================
def txt(value: str, size: int = 14, weight=ft.FontWeight.W_500, color: str = TEXT, **kw) -> ft.Text:
    return ft.Text(value, size=size, weight=weight, color=color, **kw)


def card(content: ft.Control, padding=18, radius=20, bgcolor=SURFACE, border_color=None, **kw) -> ft.Container:
    return ft.Container(
        content=content, padding=padding, border_radius=radius, bgcolor=bgcolor,
        border=ft.Border.all(1, border_color or alpha("#FFFFFF", 0.06)), **kw)


def pill(label: str, color: str, size: int = 11) -> ft.Container:
    return ft.Container(
        content=txt(label, size, ft.FontWeight.W_700, color),
        padding=ft.Padding.symmetric(horizontal=9, vertical=4), border_radius=999, bgcolor=alpha(color, 0.14))


def primary_button(label: str, icon, on_click, color: str = CRIMSON) -> ft.FilledButton:
    return ft.FilledButton(
        content=txt(label, 15, ft.FontWeight.W_700), icon=icon, on_click=on_click, expand=True,
        style=ft.ButtonStyle(bgcolor=color, color=TEXT, shape=ft.RoundedRectangleBorder(radius=16),
                             padding=ft.Padding.symmetric(horizontal=20, vertical=16)))


def ghost_button(label: str, icon, on_click, color: str = TEXT) -> ft.OutlinedButton:
    return ft.OutlinedButton(
        content=txt(label, 14, ft.FontWeight.W_600, color), icon=icon, icon_color=color, on_click=on_click,
        style=ft.ButtonStyle(side=ft.BorderSide(1, alpha(color, 0.35)), shape=ft.RoundedRectangleBorder(radius=16),
                             padding=ft.Padding.symmetric(horizontal=18, vertical=14)))


def dialog_field(hint: str, **kw) -> ft.TextField:
    return ft.TextField(
        hint_text=hint, filled=True, fill_color=SURFACE_LO, border_radius=14, text_size=15,
        border_color=alpha("#FFFFFF", 0.12), focused_border_color=CRIMSON, color=TEXT, **kw)


# =============================================================================
# 7. TAB 1 - DASHBOARD
# =============================================================================
class DashboardView:
    RINGS = [  # key, label, colour, diameter (outer -> inner)
        ("cal", "Calories", CRIMSON, 240), ("protein", "Protein", AMBER, 200),
        ("carbs", "Carbs", CARB_C, 160), ("fats", "Fats", FAT_C, 120),
    ]

    def __init__(self, app: "MacroLens") -> None:
        self.app, self.page, self.store = app, app.page, app.store
        self._token = 0
        self.shown = {k: 0.0 for k in ("cal", "protein", "carbs", "fats", "water")}

        # header -------------------------------------------------------------
        self.date_text = txt("", 13, color=TEXT_DIM)
        self.sim_btn = ft.IconButton(
            icon=ft.Icons.NOTIFICATIONS_ACTIVE_ROUNDED, icon_color=TEXT_DIM, tooltip="Preview the evening protein alert",
            on_click=self.toggle_simulation)
        header = ft.Row(
            [ft.Column([txt("Today", 30, ft.FontWeight.W_900), self.date_text], spacing=0, expand=True), self.sim_btn],
            vertical_alignment=ft.CrossAxisAlignment.CENTER)

        # protein alert ------------------------------------------------------
        self.alert_text = txt("", 13, color=TEXT_DIM)
        self.alert_box = ft.Container(
            visible=False, padding=14, border_radius=18, animate_opacity=300,
            gradient=ft.LinearGradient(begin=ft.Alignment.CENTER_LEFT, end=ft.Alignment.CENTER_RIGHT,
                                       colors=[alpha(AMBER, 0.28), alpha(CRIMSON, 0.16)]),
            border=ft.Border.all(1, alpha(AMBER, 0.5)),
            content=ft.Row(
                [ft.Icon(ft.Icons.BOLT_ROUNDED, color=AMBER, size=26),
                 ft.Column([txt("Protein is still short", 14, ft.FontWeight.W_800), self.alert_text],
                           spacing=2, expand=True),
                 ft.IconButton(icon=ft.Icons.CLOSE_ROUNDED, icon_size=18, icon_color=TEXT_DIM,
                               on_click=self.dismiss_alert)],
                vertical_alignment=ft.CrossAxisAlignment.CENTER))
        self._dismissed_on: Optional[date] = None
        self.simulate_evening = False

        # nested rings -------------------------------------------------------
        self.rings: dict[str, ft.ProgressRing] = {}
        ring_controls: list[ft.Control] = []
        for key, _label, color, size in self.RINGS:
            ring = ft.ProgressRing(value=0.0, width=size, height=size, stroke_width=14, color=color,
                                   bgcolor=alpha(color, 0.13), stroke_cap=ft.StrokeCap.ROUND)
            self.rings[key] = ring
            ring_controls.append(ring)
        self.center_num = txt("0", 28, ft.FontWeight.W_900)
        self.center_lbl = txt("kcal left", 11, color=TEXT_DIM)
        ring_controls.append(ft.Container(
            width=240, height=240, alignment=ft.Alignment.CENTER,
            content=ft.Column([self.center_num, self.center_lbl], spacing=0, tight=True,
                              horizontal_alignment=ft.CrossAxisAlignment.CENTER)))
        ring_stack = ft.Stack(ring_controls, width=240, height=240, alignment=ft.Alignment.CENTER)

        # legend tiles -------------------------------------------------------
        self.tile_text: dict[str, ft.Text] = {}
        tiles = []
        for key, label, color, _ in self.RINGS:
            self.tile_text[key] = txt("", 15, ft.FontWeight.W_800)
            tiles.append(ft.Container(
                expand=1, padding=12, border_radius=16, bgcolor=SURFACE_LO,
                content=ft.Column([
                    ft.Row([ft.Container(width=8, height=8, border_radius=4, bgcolor=color),
                            txt(label, 12, ft.FontWeight.W_600, TEXT_DIM)], spacing=6),
                    self.tile_text[key]], spacing=4)))
        legend = ft.Column([ft.Row(tiles[:2], spacing=10), ft.Row(tiles[2:], spacing=10)], spacing=10)

        ring_card = card(
            ft.Column([ft.Container(ring_stack, alignment=ft.Alignment.CENTER), legend], spacing=20,
                      horizontal_alignment=ft.CrossAxisAlignment.STRETCH),
            padding=20, radius=28,
            shadow=ft.BoxShadow(blur_radius=40, spread_radius=-12, color=alpha(CRIMSON, 0.30), offset=ft.Offset(0, 14)))

        # protein hero bar ---------------------------------------------------
        self.p_now = txt("0 g", 34, ft.FontWeight.W_900)
        self.p_of = txt("of 160 g", 14, color=TEXT_DIM)
        self.p_pct = pill("0%", AMBER, 13)
        self.p_bar = ft.ProgressBar(value=0.0, bar_height=20, border_radius=10, color=AMBER,
                                    bgcolor=alpha(AMBER, 0.14))
        self.p_foot = txt("", 13, color=TEXT_DIM)
        protein_card = card(
            ft.Column([
                ft.Row([ft.Icon(ft.Icons.FITNESS_CENTER_ROUNDED, color=AMBER, size=20),
                        txt("Protein target", 15, ft.FontWeight.W_800), ft.Container(expand=True), self.p_pct]),
                ft.Row([self.p_now, self.p_of], vertical_alignment=ft.CrossAxisAlignment.END, spacing=8),
                self.p_bar, self.p_foot], spacing=12),
            padding=20, radius=24, border_color=alpha(AMBER, 0.35))

        # water --------------------------------------------------------------
        self.w_now = txt("0 ml", 22, ft.FontWeight.W_900)
        self.w_of = txt("of 3,000 ml", 13, color=TEXT_DIM)
        self.w_bar = ft.ProgressBar(value=0.0, bar_height=12, border_radius=6, color=WATER_C,
                                    bgcolor=alpha(WATER_C, 0.14))

        def water_btn(label: str, ml: int) -> ft.FilledButton:
            return ft.FilledButton(
                content=txt(label, 13, ft.FontWeight.W_700, WATER_C), on_click=lambda e, m=ml: self.add_water(m),
                style=ft.ButtonStyle(bgcolor=alpha(WATER_C, 0.14), shape=ft.RoundedRectangleBorder(radius=14),
                                     padding=ft.Padding.symmetric(horizontal=16, vertical=12)))

        water_card = card(
            ft.Column([
                ft.Row([ft.Icon(ft.Icons.WATER_DROP_ROUNDED, color=WATER_C, size=20),
                        txt("Water", 15, ft.FontWeight.W_800), ft.Container(expand=True),
                        ft.Row([self.w_now, self.w_of], spacing=6, vertical_alignment=ft.CrossAxisAlignment.END)]),
                self.w_bar,
                ft.Row([water_btn("+250 ml", 250), water_btn("+500 ml", 500),
                        ft.IconButton(icon=ft.Icons.UNDO_ROUNDED, icon_color=TEXT_DIM, tooltip="Remove 250 ml",
                                      on_click=lambda e: self.add_water(-250))], spacing=8)], spacing=14),
            padding=18, radius=24)

        self.view = ft.Column(
            [header, self.alert_box, ring_card, protein_card, water_card], spacing=16, scroll=ft.ScrollMode.AUTO,
            expand=True, horizontal_alignment=ft.CrossAxisAlignment.STRETCH)
        self.root = ft.Container(content=self.view, padding=ft.Padding.only(left=18, right=18, top=8, bottom=12),
                                 expand=True)

    # ---- alert ------------------------------------------------------------
    def alert_message(self) -> Optional[str]:
        t = self.store.targets
        if t.protein <= 0 or (self._dismissed_on == date.today()):
            return None
        late = self.simulate_evening or datetime.now().hour >= EVENING_HOUR
        left = t.protein - self.store.totals_on(date.today())["protein"]
        if not late or left <= 0.5:
            return None
        tip = "A chicken breast or paneer plate covers it." if left > 40 else "A shake or Greek yogurt closes the gap."
        return f"{fmt(left)} g to go before the day ends. {tip}"

    def update_alert(self) -> None:
        msg = self.alert_message()
        self.alert_box.visible = msg is not None
        if msg:
            self.alert_text.value = msg
        self.page.update()

    def dismiss_alert(self, e=None) -> None:
        self._dismissed_on = date.today()
        self.update_alert()

    def toggle_simulation(self, e=None) -> None:
        self.simulate_evening = not self.simulate_evening
        self._dismissed_on = None
        self.sim_btn.icon_color = AMBER if self.simulate_evening else TEXT_DIM
        self.app.snack("Evening alert preview on" if self.simulate_evening else "Evening alert preview off")
        self.update_alert()

    # ---- data -> UI -------------------------------------------------------
    def add_water(self, ml: int) -> None:
        self.store.add_water(date.today(), ml)
        self.refresh()

    def refresh(self, replay: bool = False) -> None:
        """Recompute everything from the store. `replay` restarts the fill animation from empty."""
        t, today = self.store.targets, date.today()
        tot = self.store.totals_on(today)
        water = self.store.water_on(today)
        now = datetime.now()
        self.date_text.value = f"{now.strftime('%A')}, {now.day} {now.strftime('%B')}"

        left = t.calories - tot["cal"]
        if left >= 0:
            self.center_num.value, self.center_lbl.value, self.center_num.color = fmt(left), "kcal left", TEXT
        else:
            self.center_num.value, self.center_lbl.value, self.center_num.color = f"+{fmt(-left)}", "kcal over", CRIMSON
        self.tile_text["cal"].value = f"{fmt(tot['cal'])} / {fmt(t.calories)} kcal"
        self.tile_text["protein"].value = f"{fmt(tot['protein'])} / {fmt(t.protein)} g"
        self.tile_text["carbs"].value = f"{fmt(tot['carbs'])} / {fmt(t.carbs)} g"
        self.tile_text["fats"].value = f"{fmt(tot['fats'])} / {fmt(t.fats)} g"

        pct = 0 if t.protein <= 0 else int(round(100 * num(tot["protein"]) / t.protein))
        self.p_now.value, self.p_of.value = f"{fmt(tot['protein'])} g", f"of {fmt(t.protein)} g"
        self.p_pct.content.value = f"{pct}%"
        togo = t.protein - tot["protein"]
        self.p_foot.value = f"{fmt(togo)} g to go" if togo > 0.5 else "Target hit. Nice work."
        self.p_foot.color = TEXT_DIM if togo > 0.5 else FAT_C
        self.w_now.value, self.w_of.value = f"{fmt(water)} ml", f"of {fmt(t.water)} ml"

        goal = {
            "cal": ratio(tot["cal"], t.calories), "protein": ratio(tot["protein"], t.protein),
            "carbs": ratio(tot["carbs"], t.carbs), "fats": ratio(tot["fats"], t.fats),
            "water": ratio(water, t.water),
        }
        start = {k: 0.0 for k in goal} if replay else dict(self.shown)
        self._token += 1
        self.update_alert()
        self.page.run_task(self._tween, self._token, start, goal)

    def _apply(self, fractions: dict) -> None:
        for key in ("cal", "protein", "carbs", "fats"):
            self.rings[key].value = fractions[key]
        self.p_bar.value = fractions["protein"]
        self.w_bar.value = fractions["water"]

    async def _tween(self, token: int, start: dict, goal: dict, steps: int = 16) -> None:
        """Ease-out fill for rings and bars. A newer refresh cancels an older tween."""
        for i in range(1, steps + 1):
            if token != self._token:
                return
            t = 1 - (1 - i / steps) ** 3
            frame = {k: start[k] + (goal[k] - start[k]) * t for k in goal}
            self._apply(frame)
            self.shown = frame
            self.page.update()
            await asyncio.sleep(0.022)
        self.shown = dict(goal)

    def on_show(self) -> None:
        self.refresh(replay=True)


# =============================================================================
# 8. TAB 2 - AI CAMERA SCANNER
# =============================================================================
class ScannerView:
    FRAME_H = 330

    def __init__(self, app: "MacroLens") -> None:
        self.app, self.page, self.store, self.ai = app, app.page, app.store, app.ai
        self.pending: Optional[Analysis] = None
        self.busy = False
        self.picker = ft.FilePicker()
        self.page.services.append(self.picker)

        # viewfinder ---------------------------------------------------------
        self.idle_layer = ft.Container(
            left=0, right=0, top=0, bottom=0, alignment=ft.Alignment.CENTER,
            content=ft.Column([
                ft.Icon(ft.Icons.CENTER_FOCUS_STRONG_ROUNDED, size=54, color=alpha(AMBER, 0.9)),
                txt("Frame your plate", 17, ft.FontWeight.W_800),
                txt("Tap to take or choose a photo", 13, color=TEXT_DIM)],
                spacing=6, horizontal_alignment=ft.CrossAxisAlignment.CENTER, tight=True))
        self.preview_layer = ft.Container(left=0, right=0, top=0, bottom=0, visible=False)
        self.scan_line = ft.Container(
            left=18, right=18, top=20, height=3, border_radius=2, visible=False,
            animate_position=ft.Animation(850, ft.AnimationCurve.EASE_IN_OUT),
            gradient=ft.LinearGradient(begin=ft.Alignment.CENTER_LEFT, end=ft.Alignment.CENTER_RIGHT,
                                       colors=[alpha(CRIMSON, 0.0), CRIMSON, AMBER, alpha(AMBER, 0.0)]),
            shadow=ft.BoxShadow(blur_radius=18, color=alpha(CRIMSON, 0.9)))
        stack = ft.Stack(
            [self.idle_layer, self.preview_layer, self.scan_line,
             self._bracket("tl", CRIMSON), self._bracket("tr", AMBER),
             self._bracket("bl", AMBER), self._bracket("br", CRIMSON)])
        inner = ft.Container(content=stack, height=self.FRAME_H, border_radius=26, bgcolor="#151515",
                             clip_behavior=ft.ClipBehavior.ANTI_ALIAS, ink=True, on_click=self.pick)
        self.frame = ft.Container(
            content=inner, padding=2, border_radius=28,
            gradient=ft.LinearGradient(begin=ft.Alignment.TOP_LEFT, end=ft.Alignment.BOTTOM_RIGHT,
                                       colors=[CRIMSON, AMBER]),
            shadow=ft.BoxShadow(blur_radius=36, spread_radius=-8, color=alpha(CRIMSON, 0.4), offset=ft.Offset(0, 12)))

        self.status = txt("Good light and a top-down angle give the best read.", 13, color=TEXT_DIM,
                          text_align=ft.TextAlign.CENTER)
        self.busy_bar = ft.ProgressBar(visible=False, color=AMBER, bgcolor=alpha(AMBER, 0.14), bar_height=4,
                                       border_radius=2)
        self.result_host = ft.Container(visible=False, animate_opacity=250)

        actions = ft.Row([
            primary_button("Scan a meal", ft.Icons.ADD_A_PHOTO_ROUNDED, self.pick),
            ft.Container(ghost_button("Type it", ft.Icons.KEYBOARD_ROUNDED, self.open_manual, TEXT))], spacing=10)

        self.view = ft.Column(
            [ft.Column([txt("Scan a meal", 30, ft.FontWeight.W_900),
                        txt("Snap it. Gemini estimates calories and macros.", 13, color=TEXT_DIM)], spacing=0),
             self.frame, self.busy_bar, self.status, actions, self.result_host],
            spacing=16, scroll=ft.ScrollMode.AUTO, expand=True, horizontal_alignment=ft.CrossAxisAlignment.STRETCH)
        self.root = ft.Container(content=self.view, padding=ft.Padding.only(left=18, right=18, top=8, bottom=12),
                                 expand=True)

    def on_show(self) -> None:
        if not self.busy and self.pending is None:
            self.status.value = "Good light and a top-down angle give the best read."
            self.page.update()

    @staticmethod
    def _bracket(pos: str, color: str) -> ft.Container:
        side = ft.BorderSide(3, color)
        v, h = pos[0], pos[1]
        border = ft.Border(top=side if v == "t" else None, bottom=side if v == "b" else None,
                           left=side if h == "l" else None, right=side if h == "r" else None)
        radius = ft.BorderRadius(top_left=16 if pos == "tl" else 0, top_right=16 if pos == "tr" else 0,
                                 bottom_left=16 if pos == "bl" else 0, bottom_right=16 if pos == "br" else 0)
        placement = {("top" if v == "t" else "bottom"): 14, ("left" if h == "l" else "right"): 14}
        return ft.Container(width=38, height=38, border=border, border_radius=radius, **placement)

    # ---- capture ----------------------------------------------------------
    async def pick(self, e=None) -> None:
        if self.busy:
            return
        try:
            files = await self.picker.pick_files(
                dialog_title="Choose a meal photo", file_type=ft.FilePickerFileType.IMAGE,
                allow_multiple=False, with_data=bool(self.page.web))
        except Exception as ex:
            self.app.snack(f"Couldn't open the photo picker: {ex}")
            return
        if not files:
            return
        f = files[0]
        raw: Optional[bytes] = None
        if getattr(f, "path", None):
            try:
                raw = await asyncio.to_thread(Path(f.path).read_bytes)
            except Exception:
                raw = None
        if raw is None and getattr(f, "bytes", None):
            raw = bytes(f.bytes)
        if not raw:
            self.app.snack("That image couldn't be read. Try another photo.")
            return
        await self.analyze(raw)

    async def analyze(self, raw: bytes) -> None:
        self.set_busy(True, "Reading your plate...")
        self.result_host.visible = False
        image, mime = await asyncio.to_thread(prepare_image, raw)
        self.preview_layer.content = ft.Image(
            src=base64.b64encode(image).decode("ascii"), fit=ft.BoxFit.COVER,
            error_content=txt("Preview unavailable for this format", 13, color=TEXT_DIM))
        self.preview_layer.visible = True
        self.idle_layer.visible = False
        self.page.update()
        result = await asyncio.to_thread(self.ai.analyze_image, image, mime)
        self.set_busy(False, "")
        self.pending = result
        self.render_result()
        if result.needs_review:
            self.open_review(manual=False)

    def set_busy(self, busy: bool, message: str) -> None:
        self.busy = busy
        self.busy_bar.visible = busy
        self.scan_line.visible = busy
        if message or not busy:
            self.status.value = message or "Tap the frame to scan another meal."
        self.page.update()
        if busy:
            self.page.run_task(self._scan_animation)

    async def _scan_animation(self) -> None:
        down = True
        while self.busy:
            self.scan_line.top = self.FRAME_H - 26 if down else 20
            down = not down
            self.page.update()
            await asyncio.sleep(0.9)

    # ---- result card --------------------------------------------------------
    def render_result(self) -> None:
        a = self.pending
        if a is None:
            self.result_host.visible = False
            self.page.update()
            return
        pct = int(round(a.confidence * 100))
        conf_color = FAT_C if a.confidence >= 0.85 else AMBER if a.confidence >= LOW_CONFIDENCE else CRIMSON

        def stat(value: str, label: str, color: str) -> ft.Container:
            return ft.Container(
                expand=1, padding=ft.Padding.symmetric(horizontal=8, vertical=12), border_radius=16, bgcolor=SURFACE_LO,
                content=ft.Column([txt(value, 20, ft.FontWeight.W_900, color), txt(label, 11, color=TEXT_DIM)],
                                  spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER))

        badges = [pill("Corrected" if a.corrected else f"{pct}% match", conf_color)]
        if a.origin == "estimate":
            badges.append(pill("Estimate", TEXT_DIM))
        body = [
            ft.Row([ft.Text(glyph(a.dish), size=30),
                    ft.Column([txt(a.dish, 20, ft.FontWeight.W_800, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                               ft.Row(badges, spacing=6)], spacing=4, expand=True)], spacing=12),
            ft.Row([stat(fmt(a.calories), "kcal", CRIMSON), stat(f"{fmt(a.protein)} g", "protein", AMBER),
                    stat(f"{fmt(a.carbs)} g", "carbs", CARB_C), stat(f"{fmt(a.fats)} g", "fats", FAT_C)], spacing=8),
        ]
        if a.note:
            body.append(txt(a.note, 12, color=TEXT_DIM))
        body.append(ft.Row([primary_button("Log this meal", ft.Icons.CHECK_ROUNDED, self.log_meal)]))
        body.append(ft.Row([ft.TextButton(
            content=txt("Not right? Correct the dish", 13, ft.FontWeight.W_600, AMBER),
            on_click=lambda e: self.open_review(manual=False))], alignment=ft.MainAxisAlignment.CENTER))
        self.result_host.content = card(ft.Column(body, spacing=14), padding=18, radius=24,
                                        border_color=alpha(conf_color, 0.4))
        self.result_host.visible = True
        self.status.value = "Check the numbers, then log the meal."
        self.page.update()

    def log_meal(self, e=None) -> None:
        a = self.pending
        if a is None:
            return
        src = SRC_CORRECTED if a.corrected else SRC_ESTIMATE if a.origin == "estimate" else SRC_AI
        self.store.add_meal(Meal(uuid.uuid4().hex[:8], int(time.time()), a.dish, a.calories, a.protein, a.carbs,
                                 a.fats, a.confidence, src))
        self.reset()
        self.app.snack(f"Logged {a.dish}")
        self.app.goto(0)

    def reset(self) -> None:
        self.pending = None
        self.preview_layer.visible = False
        self.preview_layer.content = None
        self.idle_layer.visible = True
        self.result_host.visible = False
        self.status.value = "Good light and a top-down angle give the best read."

    # ---- correction modal ---------------------------------------------------
    def open_manual(self, e=None) -> None:
        if not self.busy:
            self.open_review(manual=True)

    def open_review(self, manual: bool) -> None:
        a = self.pending
        if manual:
            message: ft.Control = txt("Type the exact dish name and we'll look up its macros.", 14, color=TEXT_DIM)
            title, confirm_label = "Type a dish", "Look up macros"
        else:
            message = ft.Text(spans=[
                ft.TextSpan("We identified this as ", style=ft.TextStyle(color=TEXT_DIM)),
                ft.TextSpan(a.dish, style=ft.TextStyle(color=AMBER, weight=ft.FontWeight.W_800)),
                ft.TextSpan(". If this is incorrect, type the exact dish name below.",
                            style=ft.TextStyle(color=TEXT_DIM))], size=14)
            title, confirm_label = "Check this dish", "Update macros"

        field = dialog_field("e.g. Chicken momo, 10 pieces", autofocus=True)
        problem = txt("", 12, color=CRIMSON)
        progress = ft.ProgressBar(visible=False, color=AMBER, bgcolor=alpha(AMBER, 0.14), bar_height=4)
        dlg = ft.AlertDialog(modal=True, bgcolor=SURFACE, shape=ft.RoundedRectangleBorder(radius=26))

        def close() -> None:
            dlg.open = False
            dlg.update()

        def keep(e=None) -> None:
            close()
            if not manual:
                self.render_result()

        async def confirm(e=None) -> None:
            name = (field.value or "").strip()
            if not name:
                if manual:
                    problem.value = "Enter a dish name first."
                    dlg.update()
                    return
                keep()
                return
            if not manual and name.lower() == a.dish.strip().lower():
                keep()
                return
            problem.value, progress.visible = "", True
            confirm_btn.disabled = keep_btn.disabled = True
            dlg.update()
            result = await asyncio.to_thread(self.ai.lookup_dish, name)
            result.corrected = True
            result.confidence = max(result.confidence, 0.9)
            self.pending = result
            close()
            self.render_result()
            self.app.snack(f"Macros updated for {name}")

        field.on_submit = confirm
        keep_btn = ft.TextButton(content=txt("Cancel" if manual else "Keep this", 14, ft.FontWeight.W_600, TEXT_DIM),
                                 on_click=keep)
        confirm_btn = ft.FilledButton(
            content=txt(confirm_label, 14, ft.FontWeight.W_700), on_click=confirm,
            style=ft.ButtonStyle(bgcolor=CRIMSON, color=TEXT, shape=ft.RoundedRectangleBorder(radius=14)))
        header = [ft.Icon(ft.Icons.MANAGE_SEARCH_ROUNDED, color=AMBER), txt(title, 20, ft.FontWeight.W_800)]
        if not manual:
            header.append(pill(f"{int(round(a.confidence * 100))}%", CRIMSON))
        dlg.title = ft.Row(header, spacing=10)
        dlg.content = ft.Container(width=340, content=ft.Column(
            [message, field, problem, progress], tight=True, spacing=12))
        dlg.actions = [keep_btn, confirm_btn]
        dlg.actions_alignment = ft.MainAxisAlignment.END
        self.page.show_dialog(dlg)


# =============================================================================
# 9. TAB 3 - HISTORY LOG
# =============================================================================
class HistoryView:
    MAX_ROWS = 250

    def __init__(self, app: "MacroLens") -> None:
        self.app, self.page, self.store = app, app.page, app.store
        self.list = ft.ListView(spacing=10, expand=True, padding=ft.Padding.only(left=18, right=18, bottom=16))
        self.root = ft.Container(
            expand=True, padding=ft.Padding.only(top=8),
            content=ft.Column([
                ft.Container(padding=ft.Padding.symmetric(horizontal=18), content=ft.Column(
                    [txt("History", 30, ft.FontWeight.W_900), txt("Everything you've logged, newest first.", 13,
                                                                 color=TEXT_DIM)], spacing=0)),
                self.list], spacing=12, expand=True))

    def on_show(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        meals = sorted(self.store.meals, key=lambda m: m.ts, reverse=True)[: self.MAX_ROWS]
        rows: list[ft.Control] = []
        if not meals:
            rows.append(card(ft.Column([
                ft.Text("🍽️", size=40), txt("Nothing logged yet", 17, ft.FontWeight.W_800),
                txt("Scan your first meal and it will show up here.", 13, color=TEXT_DIM),
                ft.Row([primary_button("Scan a meal", ft.Icons.ADD_A_PHOTO_ROUNDED, lambda e: self.app.goto(1))])],
                spacing=8, horizontal_alignment=ft.CrossAxisAlignment.CENTER), padding=24, radius=24))
        current: Optional[date] = None
        for m in meals:
            d = m.when.date()
            if d != current:
                current = d
                tot = self.store.totals_on(d)
                rows.append(ft.Container(
                    padding=ft.Padding.only(top=10, bottom=2), content=ft.Row([
                        txt(day_label(d), 17, ft.FontWeight.W_800, expand=True),
                        ft.Column([txt(f"{fmt(tot['cal'])} kcal", 13, ft.FontWeight.W_700),
                                   txt(f"{fmt(tot['protein'])} g protein", 11, color=TEXT_DIM)],
                                  spacing=0, horizontal_alignment=ft.CrossAxisAlignment.END)])))
            rows.append(self._meal_card(m))
        self.list.controls = rows
        self.page.update()

    def _meal_card(self, m: Meal) -> ft.Control:
        thumb = ft.Container(
            width=54, height=54, border_radius=16, alignment=ft.Alignment.CENTER, content=ft.Text(glyph(m.name), size=26),
            gradient=ft.LinearGradient(begin=ft.Alignment.TOP_LEFT, end=ft.Alignment.BOTTOM_RIGHT,
                                       colors=[alpha(CRIMSON, 0.28), alpha(AMBER, 0.18)]))
        badges = ft.Row([
            pill(f"{fmt(m.calories)} kcal", CRIMSON), pill(f"P {fmt(m.protein)} g", AMBER),
            pill(f"C {fmt(m.carbs)} g", CARB_C), pill(f"F {fmt(m.fats)} g", FAT_C)], spacing=6, wrap=True, run_spacing=6)
        return card(ft.Row([
            thumb,
            ft.Column([
                ft.Row([txt(m.name, 15, ft.FontWeight.W_800, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, expand=True),
                        ft.IconButton(icon=ft.Icons.CLOSE_ROUNDED, icon_size=16, icon_color=TEXT_MUTE, tooltip="Remove",
                                      on_click=lambda e, mid=m.id: self.delete(mid))],
                       spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Row([txt(fmt_time(m.when), 12, color=TEXT_DIM), txt(SRC_LABEL.get(m.src, "Logged"), 12,
                                                                        ft.FontWeight.W_600, TEXT_MUTE)], spacing=10),
                badges], spacing=6, expand=True)],
            spacing=12, vertical_alignment=ft.CrossAxisAlignment.START), padding=12, radius=20)

    def delete(self, meal_id: str) -> None:
        removed = self.store.remove_meal(meal_id)
        if not removed:
            return

        def undo(e=None) -> None:
            self.store.add_meal(removed)
            self.refresh()
            self.app.dash.refresh()

        self.refresh()
        self.app.dash.refresh()
        self.app.snack(f"Removed {removed.name}", action="Undo", on_action=undo)


# =============================================================================
# 10. TAB 4 - PROFILE / TARGET SETTINGS
# =============================================================================
class TargetRow:
    """One target: label, numeric box and athletic slider kept in sync."""

    def __init__(self, view: "ProfileView", key: str, label: str, color: str, lo: int, hi: int, step: int, unit: str):
        self.view, self.key, self.lo, self.hi, self.step, self.unit = view, key, lo, hi, step, unit
        self.field = ft.TextField(
            width=96, dense=True, text_align=ft.TextAlign.RIGHT, keyboard_type=ft.KeyboardType.NUMBER,
            input_filter=ft.NumbersOnlyInputFilter(), filled=True, fill_color=SURFACE_LO, border_radius=12,
            border_color=alpha("#FFFFFF", 0.1), focused_border_color=color, text_size=16, color=TEXT,
            on_change=self._typed, on_blur=self._blur)
        self.slider = ft.Slider(min=lo, max=hi, divisions=max(1, (hi - lo) // step), value=lo, active_color=color,
                                inactive_color=alpha(color, 0.18), thumb_color=color, label="{value}",
                                on_change=self._slid)
        self.error = txt("", 12, color=CRIMSON, visible=False)
        self.control = ft.Column([
            ft.Row([ft.Container(width=10, height=10, border_radius=5, bgcolor=color),
                    txt(label, 15, ft.FontWeight.W_700, expand=True), self.field,
                    ft.Container(width=30, content=txt(unit, 12, color=TEXT_DIM))], spacing=8),
            self.slider, self.error], spacing=0)

    def set_value(self, v: int) -> None:
        self.field.value = str(int(v))
        self.slider.value = max(self.lo, min(self.hi, int(v)))
        self._flag(None)

    def value(self) -> Optional[int]:
        v = num(self.field.value, None)
        return None if v is None else int(round(v))

    def _flag(self, message: Optional[str]) -> None:
        self.error.value, self.error.visible = message or "", bool(message)
        self.field.border_color = CRIMSON if message else alpha("#FFFFFF", 0.1)

    def validate(self) -> Optional[int]:
        v = self.value()
        if v is None:
            self._flag("Enter a number.")
            return None
        if not (self.lo <= v <= self.hi):
            self._flag(f"Use {self.lo:,} to {self.hi:,} {self.unit}.")
            return None
        self._flag(None)
        return v

    def _typed(self, e) -> None:
        v = self.value()
        if v is not None and self.lo <= v <= self.hi:
            self.slider.value = v
            self._flag(None)
        self.view.update_math()

    def _blur(self, e) -> None:
        self.validate()
        self.view.page.update()

    def _slid(self, e) -> None:
        v = int(round(num(self.slider.value, self.lo) / self.step) * self.step)
        self.field.value = str(v)
        self._flag(None)
        self.view.update_math()


class ProfileView:
    SPECS = [  # key, label, colour, min, max, step, unit
        ("calories", "Daily calories", CRIMSON, 800, 6000, 50, "kcal"),
        ("protein", "Protein", AMBER, 20, 400, 5, "g"),
        ("carbs", "Carbs", CARB_C, 20, 800, 5, "g"),
        ("fats", "Fats", FAT_C, 10, 300, 5, "g"),
        ("water", "Water", WATER_C, 500, 8000, 50, "ml"),
    ]

    def __init__(self, app: "MacroLens") -> None:
        self.app, self.page, self.store, self.ai = app, app.page, app.store, app.ai
        self.rows = {s[0]: TargetRow(self, *s) for s in self.SPECS}
        self.math_text = txt("", 12, color=TEXT_DIM)
        self.key_status = pill("", AMBER)
        self.key_hint = txt("", 12, color=TEXT_DIM)

        ai_card = card(ft.Column([
            ft.Row([ft.Icon(ft.Icons.AUTO_AWESOME_ROUNDED, color=AMBER, size=20),
                    txt("AI connection", 15, ft.FontWeight.W_800, expand=True), self.key_status]),
            self.key_hint,
            ft.Row([ghost_button("Set API key", ft.Icons.KEY_ROUNDED, lambda e: self.app.open_key_dialog(), AMBER)])],
            spacing=10), padding=18, radius=22)

        targets_card = card(ft.Column(
            [self.rows[s[0]].control for s in self.SPECS] + [self.math_text], spacing=10), padding=18, radius=22)

        self.view = ft.Column(
            [ft.Column([txt("Targets", 30, ft.FontWeight.W_900),
                        txt("Set the numbers you train and eat to.", 13, color=TEXT_DIM)], spacing=0),
             targets_card,
             ft.Row([primary_button("Save targets", ft.Icons.SAVE_ROUNDED, self.save)]),
             ai_card],
            spacing=16, scroll=ft.ScrollMode.AUTO, expand=True, horizontal_alignment=ft.CrossAxisAlignment.STRETCH)
        self.root = ft.Container(content=self.view, padding=ft.Padding.only(left=18, right=18, top=8, bottom=12),
                                 expand=True)
        self.load()

    def on_show(self) -> None:
        self.load()
        self.page.update()

    def load(self) -> None:
        t = self.store.targets
        for key, row in self.rows.items():
            row.set_value(getattr(t, key))
        self.update_math()
        self.refresh_key_status()

    def update_math(self) -> None:
        vals = {k: r.value() for k, r in self.rows.items()}
        if any(v is None for v in vals.values()):
            return
        implied = 4 * vals["protein"] + 4 * vals["carbs"] + 9 * vals["fats"]
        gap = implied - vals["calories"]
        off = abs(gap) / max(vals["calories"], 1)
        self.math_text.value = (f"Your macros add up to {fmt(implied)} kcal, "
                                f"{fmt(abs(gap))} {'over' if gap > 0 else 'under'} your calorie target.")
        self.math_text.color = AMBER if off > 0.1 else TEXT_DIM
        if self.page.controls:
            self.page.update()

    def refresh_key_status(self) -> None:
        if self.ai.live:
            self.key_status.content.value, color = "Connected", FAT_C
            self.key_hint.value = ("Using GEMINI_API_KEY from the environment." if self.ai.env_key
                                   else f"Key saved on this device. Model: {self.ai.model}.")
        else:
            self.key_status.content.value, color = "Offline demo", AMBER
            self.key_hint.value = "Scans return sample estimates until you add a Gemini API key."
        self.key_status.content.color = color
        self.key_status.bgcolor = alpha(color, 0.14)

    def save(self, e=None) -> None:
        values, bad = {}, False
        for key, row in self.rows.items():
            v = row.validate()
            if v is None:
                bad = True
            else:
                values[key] = v
        if bad:
            self.page.update()
            self.app.snack("Fix the highlighted targets first.")
            return
        self.store.targets = Targets(**values)
        self.store.save()
        self.app.dash.refresh(replay=True)   # global state changed: recompute the dashboard now
        self.page.update()
        self.app.snack("Targets saved")


# =============================================================================
# 11. APP SHELL
# =============================================================================
class MacroLens:
    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self.store = StorageManager()
        self.ai = AIService(self.store)

    # ---- start-up -----------------------------------------------------------
    def start(self) -> None:
        p = self.page
        p.title = APP_NAME
        p.theme_mode = ft.ThemeMode.DARK
        p.bgcolor = BG
        p.padding = 0
        p.spacing = 0
        scheme = dict(primary=CRIMSON, secondary=AMBER, surface=SURFACE, on_surface=TEXT, on_primary=TEXT,
                      error=CRIMSON)
        p.theme = ft.Theme(color_scheme=ft.ColorScheme(**scheme))
        p.dark_theme = ft.Theme(color_scheme=ft.ColorScheme(**scheme))
        try:  # desktop preview: phone-sized window
            p.window.width, p.window.height = 400, 860
        except Exception:
            pass

        self.dash = DashboardView(self)
        self.scan = ScannerView(self)
        self.hist = HistoryView(self)
        self.prof = ProfileView(self)
        self.tabs = [self.dash, self.scan, self.hist, self.prof]

        self.key_banner = self._build_key_banner()
        # All four tabs stay mounted; switching only flips visibility + a quick fade, so nothing
        # is rebuilt (no flicker) and scroll positions survive tab changes.
        for tab in self.tabs:
            tab.root.animate_opacity = ft.Animation(220, ft.AnimationCurve.EASE_OUT)
            tab.root.visible = tab is self.dash

        p.navigation_bar = ft.NavigationBar(
            selected_index=0, bgcolor=SURFACE, indicator_color=alpha(CRIMSON, 0.22), on_change=self._on_nav,
            destinations=[
                ft.NavigationBarDestination(icon=ft.Icons.DONUT_LARGE_OUTLINED, selected_icon=ft.Icons.DONUT_LARGE_ROUNDED,
                                            label="Dashboard"),
                ft.NavigationBarDestination(icon=ft.Icons.CAMERA_ALT_OUTLINED, selected_icon=ft.Icons.CAMERA_ALT_ROUNDED,
                                            label="Scan"),
                ft.NavigationBarDestination(icon=ft.Icons.HISTORY_ROUNDED, selected_icon=ft.Icons.HISTORY_ROUNDED,
                                            label="History"),
                ft.NavigationBarDestination(icon=ft.Icons.TUNE_ROUNDED, selected_icon=ft.Icons.TUNE_ROUNDED,
                                            label="Targets"),
            ])
        p.add(ft.SafeArea(expand=True, avoid_intrusions_bottom=False,
                          content=ft.Column([self.key_banner, *[t.root for t in self.tabs]], spacing=0, expand=True)))
        self.dash.refresh(replay=True)
        p.run_task(self._notification_loop)

    async def _notification_loop(self) -> None:
        """Simulated notification checker: re-evaluates the evening protein alert every 30 s."""
        while True:
            await asyncio.sleep(30)
            try:
                self.dash.update_alert()
            except Exception:
                return  # session closed

    # ---- navigation ---------------------------------------------------------
    def _on_nav(self, e) -> None:
        self.goto(int(e.control.selected_index))

    def goto(self, index: int) -> None:
        self.page.navigation_bar.selected_index = index
        view = self.tabs[index]
        for tab in self.tabs:
            tab.root.visible = tab is view
        view.root.opacity = 0
        self.page.update()
        view.on_show()
        self.page.run_task(self._fade_in, view.root)

    async def _fade_in(self, root: ft.Control) -> None:
        await asyncio.sleep(0.03)
        root.opacity = 1
        self.page.update()

    # ---- shared UI ----------------------------------------------------------
    def snack(self, message: str, action: Optional[str] = None, on_action=None) -> None:
        self.page.show_dialog(ft.SnackBar(
            content=txt(message, 14, ft.FontWeight.W_600), bgcolor=SURFACE_HI, duration=3200,
            behavior=ft.SnackBarBehavior.FLOATING, shape=ft.RoundedRectangleBorder(radius=14),
            action=ft.SnackBarAction(label=action, on_click=on_action) if action else None))

    def _build_key_banner(self) -> ft.Container:
        return ft.Container(
            visible=not self.ai.live, margin=ft.Margin.only(left=14, right=14, top=6), padding=14, border_radius=18,
            animate_opacity=300,
            gradient=ft.LinearGradient(begin=ft.Alignment.CENTER_LEFT, end=ft.Alignment.CENTER_RIGHT,
                                       colors=[alpha(CRIMSON, 0.30), alpha(AMBER, 0.16)]),
            border=ft.Border.all(1, alpha(CRIMSON, 0.55)),
            content=ft.Row([
                ft.Icon(ft.Icons.KEY_ROUNDED, color=AMBER, size=26),
                ft.Column([txt("Add your Gemini API key", 14, ft.FontWeight.W_800),
                           txt("Until then, scans use sample estimates so you can test the app.", 12, color=TEXT_DIM)],
                          spacing=2, expand=True),
                ft.FilledButton(content=txt("Add key", 13, ft.FontWeight.W_700), on_click=lambda e: self.open_key_dialog(),
                                style=ft.ButtonStyle(bgcolor=CRIMSON, color=TEXT,
                                                     shape=ft.RoundedRectangleBorder(radius=12)))],
                vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=10))

    def open_key_dialog(self) -> None:
        field = dialog_field("Paste your Gemini API key", password=True, can_reveal_password=True, autofocus=True,
                             value=self.store.api_key)
        problem = txt("", 12, color=CRIMSON)
        dlg = ft.AlertDialog(modal=True, bgcolor=SURFACE, shape=ft.RoundedRectangleBorder(radius=26))

        def close() -> None:
            dlg.open = False
            dlg.update()

        def apply(key: str) -> None:
            self.store.api_key = key
            self.store.save()
            self.key_banner.visible = not self.ai.live
            self.prof.refresh_key_status()
            close()
            self.page.update()
            self.snack("Gemini connected" if self.ai.live else "Key removed. Using sample estimates.")

        def save(e=None) -> None:
            key = (field.value or "").strip()
            if len(key) < 20 or " " in key:
                problem.value = "That doesn't look like a valid key."
                dlg.update()
                return
            apply(key)

        dlg.title = ft.Row([ft.Icon(ft.Icons.KEY_ROUNDED, color=AMBER), txt("Gemini API key", 20, ft.FontWeight.W_800)],
                           spacing=10)
        dlg.content = ft.Container(width=340, content=ft.Column([
            txt("Create a key in Google AI Studio. It is stored only in this app's private storage on your device.",
                13, color=TEXT_DIM), field, problem], tight=True, spacing=12))
        field.on_submit = save
        dlg.actions = [
            ft.TextButton(content=txt("Cancel", 14, ft.FontWeight.W_600, TEXT_DIM), on_click=lambda e: close()),
            ft.TextButton(content=txt("Remove key", 14, ft.FontWeight.W_600, CRIMSON), on_click=lambda e: apply("")),
            ft.FilledButton(content=txt("Save key", 14, ft.FontWeight.W_700), on_click=save,
                            style=ft.ButtonStyle(bgcolor=CRIMSON, color=TEXT, shape=ft.RoundedRectangleBorder(radius=14))),
        ]
        dlg.actions_alignment = ft.MainAxisAlignment.END
        self.page.show_dialog(dlg)


def main(page: ft.Page) -> None:
    MacroLens(page).start()


if __name__ == "__main__":
    ft.run(main)
