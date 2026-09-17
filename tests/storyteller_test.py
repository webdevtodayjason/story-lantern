"""Who tells the story, and what happens when nobody can.

    python3 tests/storyteller_test.py

Offline. The fake Tiiny from fake_models.py stands in for the model-lifecycle
routes and a small chat stub stands in for the rest, so the whole of this runs
with no hardware and no network.

The bug being held down: Story Lantern 0.1.2 hard-coded one chat model id, so a
device holding two perfectly good chat models answered the first call of every
story with HTTP 404 "is not loaded" and the lamp said goodnight five seconds
after the child asked. Two people hit it the same night. What follows is the
contract that replaced it - choose from what is running, prefer Ornith, and when
there is genuinely nobody, refuse the story before any page work with something a
parent can act on.
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

HOME = tempfile.mkdtemp(prefix="lantern-storyteller-")
os.environ["TIINY_HOST"] = "127.0.0.1"
os.environ["TIINY_KEY"] = "fake"
os.environ["LANTERN_HOME"] = HOME
os.environ["PORT"] = "8460"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lantern as L  # noqa: E402
import fake_models  # noqa: E402
from fake_models import (FakeModels, ORNITH, BIG_CHAT, CHAT, CODER,  # noqa: E402
                         TTS, IMAGE, EMBED, RERANK, OCR, ASR)

PORT = 8460
PNG = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
MP3 = (b"\xff\xfb\x90\x00" + b"\x00" * 64)

MODELS = FakeModels(running=[CHAT], npu_available=16)
CALLS = []          # every job the device was asked for, in order
EVENTS = []         # every (kind, data, ui) the bus published


def fake_http(self, job):
    CALLS.append((job.kind, job.path, (job.body or {}).get("model")))
    time.sleep(0.005)
    if job.kind == "control":
        answer = MODELS.control(job.path)
        return {"ok": True} if answer is None else answer
    if job.kind == "image":
        return PNG
    if job.kind == "speech":
        return MP3
    text = json.dumps(job.body["messages"])
    if "spine" in text and "first_page" in text:
        return _chat({"title": "The Moth and the Lamp", "setting": "home",
                      "characters": [{"name": "Moth", "kind": "invented",
                                      "descriptor": "a small grey moth with dusty wings",
                                      "personality": "curious and gentle"}],
                      "spine": ["it flies", "it sleeps", "it dreams"],
                      "first_page": {"text": " ".join(["word"] * 40),
                                     "scene": "a moth beside a warm lamp"}})
    if "suspicious" in text or "cautious adult" in text:
        return _chat({"verdict": "ALLOW", "reason": "fine"})
    return _chat({"text": "The little moth settled on the warm glass and slept. " * 3,
                  "scene": "a moth on warm glass", "characters": ["Moth"]})


def _chat(obj):
    return {"choices": [{"message": {"content": json.dumps(obj)}}]}


_publish = L.EventBus.publish


def spy_publish(self, kind, data, ui=None):
    EVENTS.append((kind, data, ui))
    return _publish(self, kind, data, ui)


L.DeviceWorker._http = fake_http
L.EventBus.publish = spy_publish

L.CFG.child_name = "Maya"
L.CFG.child_age = 5
L.CFG.busy_max_wait = 10
L.CFG.prefetch_stall_s = 2
L.db_init()
LAN = L.Lantern(L.CFG)
L.Handler.lantern = LAN
HTTPD = L.LanternServer(("127.0.0.1", PORT), L.Handler)
threading.Thread(target=HTTPD.serve_forever, daemon=True).start()


def post(path, obj):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}" + path,
                                 data=json.dumps(obj).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.status, json.loads(r.read())


def get(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}" + path, timeout=20) as r:
        return json.loads(r.read())


def run_story(request, pages=2, timeout=90):
    """Post a story and wait for the producer thread to finish with it."""
    status, body = post("/api/story", {"request": request, "pages": pages})
    session = LAN.session
    deadline = time.time() + timeout
    while session.thread.is_alive() and time.time() < deadline:
        time.sleep(0.05)
    return status, body, session


def ui_frames(kind):
    return [ui[1] for _, _, ui in EVENTS if ui and ui[0] == kind]


class PickTest(unittest.TestCase):
    """pick_storyteller, against a device holding various things."""

    def pick(self, running):
        return L.pick_storyteller(LAN.worker, running=running,
                                  catalog=_catalog())

    def test_prefers_ornith_when_it_is_running(self):
        self.assertEqual(self.pick([CHAT, BIG_CHAT, ORNITH]), ORNITH)

    def test_prefers_the_newest_qwen_over_the_small_one(self):
        self.assertEqual(self.pick([CHAT, BIG_CHAT]), BIG_CHAT)

    def test_picks_the_only_chat_model_there_is(self):
        self.assertEqual(self.pick([CHAT, EMBED, TTS]), CHAT)

    def test_skips_everything_that_cannot_tell_a_story(self):
        self.assertIsNone(self.pick([CODER, EMBED, RERANK, OCR, ASR, TTS, IMAGE]))

    def test_none_when_the_device_holds_nothing(self):
        self.assertIsNone(self.pick([]))

    def test_matches_a_renamed_id_case_insensitively(self):
        # The store calls it "...A3B-Turbo" and the installed copy calls it
        # "...A3B-turbo". Comparing bytes says a loaded model is not loaded.
        self.assertTrue(L._same_model("Qwen/Qwen3.6-35B-A3B-turbo",
                                      "Qwen/Qwen3.6-35B-A3B-Turbo"))
        self.assertTrue(L._listed(["Qwen/Qwen3.6-35B-A3B-turbo"],
                                  "Qwen/Qwen3.6-35B-A3B-Turbo"))

    def test_reads_the_device_when_told_nothing(self):
        MODELS.running = [CHAT]
        self.assertEqual(L.pick_storyteller(LAN.worker), CHAT)


def _catalog():
    return {mid.lower(): MODELS._catalog_row(mid) for mid in
            fake_models.SPECS}


class StoryTest(unittest.TestCase):
    """A whole story, told by whoever is holding the device tonight."""

    def setUp(self):
        MODELS.running = [CHAT]
        MODELS.installed = list(fake_models.SPECS)
        MODELS.npu_available = 16
        MODELS.starts.clear()
        CALLS.clear()
        EVENTS.clear()

    def test_a_story_told_by_the_only_chat_model_completes(self):
        status, body, session = run_story("a story about a quiet moth")
        self.assertEqual(status, 201)
        self.assertEqual(session.status, "finished", session.status)
        self.assertEqual(session.storyteller, CHAT)
        self.assertGreaterEqual(len(session.pages), 1)
        doc = get(f"/api/story/{body['story_id']}")
        self.assertEqual(doc["storyteller"], "Qwen/Qwen3-8B")
        self.assertIsNone(doc["reason"])

    def test_every_chat_call_used_the_chosen_model(self):
        run_story("a story about a quiet moth")
        models = {m for kind, _, m in CALLS if kind == "chat"}
        self.assertEqual(models, {CHAT}, models)

    def test_state_route_reports_the_storyteller(self):
        run_story("a story about a quiet moth")
        self.assertEqual(get("/api/state")["storyteller"], CHAT)

    def test_no_storyteller_refuses_the_story_in_the_request(self):
        MODELS.running = [EMBED, RERANK, TTS]
        MODELS.npu_available = 4          # Ornith needs 50; it does not fit
        try:
            post("/api/story", {"request": "a story about a quiet moth", "pages": 2})
            self.fail("a story with no storyteller must not be accepted")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 503)
            body = json.loads(exc.read())
        self.assertEqual(body["reason"], "no_storyteller")

        # Nothing was spent on a story that could not be told.
        self.assertEqual([k for k, _, _ in CALLS if k in ("chat", "image", "speech")], [])
        self.assertNotIn(ORNITH, MODELS.starts)

        doc = get(f"/api/story/{body['story_id']}")
        self.assertEqual(doc["status"], "failed")
        self.assertEqual(doc["reason"], "no_storyteller")
        self.assertEqual(doc["pages"], [])
        self.assertIsNone(doc["storyteller"])

        # The parent sentence names what to load AND what the device is holding.
        line = doc["parent_line"]
        self.assertIn("No chat model is loaded on the Tiiny.", line)
        for mid in (EMBED, RERANK, TTS):
            self.assertIn(mid, line)

    def test_no_storyteller_shows_a_card_and_never_a_goodnight(self):
        MODELS.running = [EMBED]
        MODELS.npu_available = 4
        try:
            post("/api/story", {"request": "a story about a quiet moth", "pages": 2})
        except urllib.error.HTTPError:
            pass
        ends = ui_frames("end")
        self.assertEqual(len(ends), 1, ends)
        self.assertEqual(ends[0]["reason"], "no_storyteller")
        self.assertEqual(ends[0]["line"], L.NO_STORYTELLER_LINE)
        # No model id, no host, no exception text anywhere near the lamp.
        for _, _, ui in EVENTS:
            if ui:
                blob = json.dumps(ui[1])
                self.assertNotIn("Qwen", blob)
                self.assertNotIn("http", blob)

    def test_the_preferred_model_is_started_when_it_fits(self):
        MODELS.running = [EMBED]
        MODELS.npu_available = 60         # room for Ornith's 50
        status, body, session = run_story("a story about a quiet moth")
        self.assertEqual(status, 201)
        self.assertEqual(MODELS.starts[0], ORNITH)
        self.assertIn(ORNITH, MODELS.running)
        self.assertEqual(session.storyteller, ORNITH)
        self.assertEqual(session.status, "finished", session.status)

    def test_a_failure_after_page_one_still_says_goodnight(self):
        real_build = L.StorySession._build_page

        def break_after_the_first(self, idx):
            if idx == 0:
                return real_build(self, idx)
            raise L.DeviceError("HTTP 500 page2.text: upstream fell over")

        L.StorySession._build_page = break_after_the_first
        try:
            _, body, session = run_story("a story about a quiet moth", pages=3)
        finally:
            L.StorySession._build_page = real_build
        self.assertEqual(session.status, "failed")
        self.assertEqual(len(session.pages), 1)
        ends = ui_frames("end")
        self.assertEqual(ends[-1]["reason"], "failed")
        self.assertEqual(ends[-1]["line"], "that is enough story for tonight")

    def test_a_missing_voice_degrades_to_text(self):
        def no_voice(self, job):
            if job.kind == "speech":
                raise L.DeviceError("HTTP 404 model_not_found")
            return fake_http(self, job)

        L.DeviceWorker._http = no_voice
        try:
            _, _, session = run_story("a story about a quiet moth")
        finally:
            L.DeviceWorker._http = fake_http
        self.assertEqual(session.status, "finished", session.status)
        self.assertTrue(all(p.text for p in session.pages.values()))
        self.assertFalse(any(p.audio_path for p in session.pages.values()))
        self.assertIn("audio", [d.get("what") for k, d, _ in EVENTS
                                if k == "page.degraded"])

    def test_a_missing_illustrator_degrades_to_no_plate(self):
        def no_plates(self, job):
            if job.kind == "image":
                raise L.DeviceError("HTTP 503 model evicted")
            return fake_http(self, job)

        L.DeviceWorker._http = no_plates
        try:
            _, _, session = run_story("a story about a quiet moth")
        finally:
            L.DeviceWorker._http = fake_http
        self.assertEqual(session.status, "finished", session.status)
        self.assertFalse(any(p.image_path for p in session.pages.values()))
        self.assertTrue(all(p.audio_path for p in session.pages.values()))
        self.assertIn("image", [d.get("what") for k, d, _ in EVENTS
                                if k == "page.degraded"])


def tearDownModule():
    HTTPD.shutdown()
    LAN.shutdown()
    shutil.rmtree(HOME, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
