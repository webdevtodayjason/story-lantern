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
import re
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


class ReasonTest(unittest.TestCase):
    """One word for what went wrong, and whether the right thing can produce it.

    The lamp ships a card for a device nobody can hear. For most of this
    release that card was unreachable by an unreachable device and reachable
    by a busy one, because the retry budget flattened DeviceUnreachable into a
    plain DeviceError on its way out and LanternBusy was reading as the same
    condition. Both halves are held down here.
    """

    def test_an_unreachable_device_survives_the_retry_budget_as_itself(self):
        def dead(self, job):
            raise L.DeviceUnreachable(
                f"unreachable {job.label}: [Errno 61] Connection refused")

        job = L.DeviceJob(kind="control", path="/api/v1/models/running",
                          method="GET", label="probe", timeout=1.0)
        L.DeviceWorker._http = dead
        try:
            with self.assertRaises(L.DeviceUnreachable) as caught:
                LAN.worker._call_with_backoff(job)
        finally:
            L.DeviceWorker._http = fake_http
        self.assertEqual(L.failure_reason(caught.exception), "device_unreachable")

    def test_a_busy_device_is_not_reported_as_one_nobody_can_hear(self):
        # It answered. It said it was doing something else. That is the lantern
        # dreaming, and sending a parent out to check the Tiiny over it sends
        # them looking for a fault that is not there.
        self.assertEqual(L.failure_reason(L.LanternBusy("busy for 90s (plan)")),
                         "failed")
        self.assertEqual(L.failure_reason(L.DeviceError("HTTP 500 plan")), "failed")
        self.assertEqual(L.failure_reason(L.NoStoryteller("nobody", [])),
                         "no_storyteller")

    def test_a_device_that_swallows_the_call_is_still_unreachable(self):
        """The shape of an actual unplugged Tiiny, which is the common one.

        It does not refuse the connection, it says nothing at all, so every call
        sits until the caller gives up and the story sees a bare timeout. The
        worker is the only thing that knows the device was never reached, and
        without it this night reported itself as "something went wrong".
        """
        worker = LAN.worker
        was = worker.unreachable_since
        try:
            worker.unreachable_since = time.time()
            self.assertEqual(L.failure_reason(TimeoutError(), worker),
                             "device_unreachable")
            # An answer of any kind clears it, and then a timeout is just a
            # timeout: a device deep in an inference is not a missing one.
            worker.unreachable_since = None
            self.assertEqual(L.failure_reason(TimeoutError(), worker), "failed")
            self.assertEqual(L.failure_reason(L.LanternBusy("busy"), worker), "failed")
        finally:
            worker.unreachable_since = was

    def test_the_worker_remembers_reaching_nothing_and_forgets_on_an_answer(self):
        def dead(self, job):
            raise L.DeviceUnreachable(f"unreachable {job.label}: timed out")

        job = L.DeviceJob(kind="control", path="/api/v1/models/running",
                          method="GET", label="probe", timeout=1.0)
        L.DeviceWorker._http = dead
        try:
            with self.assertRaises(L.DeviceUnreachable):
                LAN.worker._call_with_backoff(job)
            self.assertIsNotNone(LAN.worker.unreachable_since)
        finally:
            L.DeviceWorker._http = fake_http
        LAN.worker._call_with_backoff(job)
        self.assertIsNone(LAN.worker.unreachable_since)

    def test_every_reason_has_a_line_for_the_child(self):
        for reason in ("no_storyteller", "device_unreachable", "failed"):
            line = L.CHILD_TROUBLE_LINE[reason]
            self.assertIn("grown-up", line)
            for word in ("Qwen", "Ornith", "http", "HTTP", "NPU"):
                self.assertNotIn(word, line)


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

    def test_the_lamps_own_transport_never_carries_the_model_id(self):
        """/api/state is the SSE hello and the lamp's 2.5s poll fallback.

        Anything in that body is pushed to a child's screen on every reconnect,
        and the lamp is the one screen in this product that must never show a
        model id. Who told the story is a question for /api/story/<id> and the
        parent page, which is where the answer stayed.
        """
        _, body, _ = run_story("a story about a quiet moth")
        state = get("/api/state")
        self.assertNotIn("storyteller", state)
        blob = json.dumps(state)
        for mid in (CHAT, BIG_CHAT, ORNITH):
            self.assertNotIn(mid, blob)
        self.assertEqual(get(f"/api/story/{body['story_id']}")["storyteller"], CHAT)

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

        # And it reaches the page that renders it, by the same field name.
        row = [s for s in get("/api/parent/data")["stories"]
               if s["id"] == body["story_id"]][0]
        self.assertEqual(row["reason"], "no_storyteller")
        self.assertEqual(row["parent_line"], line)

        # The lamp, meanwhile, is told nothing but that it cannot start.
        self.assertNotIn("storyteller", get("/api/state"))

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
        self.assertIs(ends[0]["started"], False)
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
        # No started flag: this story DID happen, so the lamp says goodnight to
        # it rather than raising a card over an evening that is already over.
        self.assertNotIn("started", ends[-1])
        self.assertFalse(session.failed_before_pages)
        # And a lamp that reconnects after it is not answered with a card.
        self.assertIsNone(get("/api/state")["reason"])

    def test_a_plan_that_falls_over_shows_a_card_and_not_a_goodnight(self):
        """The same night, one page earlier, and it used to read identically.

        A pre-page failure sent the storyteller line for every reason, so a
        healthy device with a broken plan call told the grown-up to go and load
        a model - and the lamp, which had no card for plain "failed", said
        Goodnight to a story that never started. That is the 22:48 symptom
        coming back through a second door.
        """
        def no_plan(self, job):
            if job.kind == "chat":
                raise L.DeviceError("HTTP 500 plan: upstream fell over")
            return fake_http(self, job)

        # A Tiiny that went missing earlier in the evening and came back must
        # not colour this failure. The reads this story makes succeed, and a
        # device that answers is a device that is there.
        LAN.worker.unreachable_since = time.time()
        L.DeviceWorker._http = no_plan
        try:
            _, body, session = run_story("a story about a quiet moth")
        finally:
            L.DeviceWorker._http = fake_http
        self.assertEqual(session.status, "failed")
        self.assertEqual(session.pages, {})
        self.assertTrue(session.failed_before_pages)
        frame = ui_frames("end")[-1]
        self.assertIs(frame["started"], False)
        self.assertEqual(frame["reason"], "failed")
        self.assertEqual(frame["line"], L.CHILD_TROUBLE_LINE["failed"])
        self.assertNotEqual(frame["line"], L.NO_STORYTELLER_LINE)
        # The storyteller was never the problem, and the parent log says so.
        doc = get(f"/api/story/{body['story_id']}")
        self.assertEqual(doc["storyteller"], CHAT)
        self.assertIn("500", doc["parent_line"])

    def test_a_device_that_never_answers_says_so(self):
        """A Tiiny that is off, unplugged or off the LAN.

        The story is refused before any page work and the reason names the
        condition, which is what the lamp's second card and the parent page
        both key on.
        """
        def gone(self, job):
            raise L.DeviceUnreachable(
                f"unreachable {job.label}: [Errno 61] Connection refused")

        manage = L.CFG.manage_tts
        L.CFG.manage_tts = False      # no 240s voice poll against a dead host
        L.DeviceWorker._http = gone
        try:
            _, body, session = run_story("a story about a quiet moth", timeout=120)
        finally:
            L.DeviceWorker._http = fake_http
            L.CFG.manage_tts = manage
        self.assertEqual(session.status, "failed")
        self.assertEqual(session.failure_reason, "device_unreachable")
        self.assertEqual(session.pages, {})
        frame = ui_frames("end")[-1]
        self.assertIs(frame["started"], False)
        self.assertEqual(frame["reason"], "device_unreachable")
        self.assertEqual(frame["line"], L.CHILD_TROUBLE_LINE["device_unreachable"])
        self.assertEqual(get(f"/api/story/{body['story_id']}")["reason"],
                         "device_unreachable")

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


def _static(name):
    with open(os.path.join(ROOT, "static", name), encoding="utf-8") as fh:
        return fh.read()


class LampPageTest(unittest.TestCase):
    """static/show.html, read as source.

    There is no browser here and no JS engine in the standard library, so these
    are contract checks over the file rather than a render, and they say so. The
    two properties they hold down are exactly the two a green Python suite kept
    proving nothing about: a story that never started raises the card, and no
    line the lamp can put on screen names a model.
    """

    SHOW = _static("show.html")

    def test_a_story_that_never_started_raises_the_card(self):
        # The flag, not the reason: "failed" arrives on both sides of the first
        # page and only one of them is an ending.
        self.assertTrue(re.search(r"data\.started\s*===\s*false\s*\)\s*\{\s*"
                                  r"showTrouble\(", self.SHOW), "the end handler "
                        "no longer routes a never-started story to the card")

    def test_the_card_is_decided_before_the_goodnight(self):
        card = self.SHOW.index("data.started === false")
        goodnight = self.SHOW.index("if (data.line) notice(data.line")
        self.assertLess(card, goodnight)

    def test_there_is_a_line_for_every_reason_the_engine_can_send(self):
        block = self.SHOW[self.SHOW.index("var TROUBLE = {"):]
        block = block[:block.index("};")]
        for reason in ("no_storyteller", "device_unreachable", "failed"):
            self.assertIn(reason + ":", block)
        for word in ("Qwen", "Ornith", "http", "HTTP", "NPU", "404"):
            self.assertNotIn(word, block)


class ParentPageTest(unittest.TestCase):
    """static/parent.html, read as source, for the same reason.

    The engine has carried storyteller, reason and parent_line on
    /api/parent/data and /api/story/<id> since the storyteller fix landed. The
    page dropped all three in a field whitelist that predates them and had no
    markup for any of them, so the one screen built to explain a bad night
    rendered nothing about it - which is the condition the two people who hit
    the 0.1.2 bug were left in.
    """

    PARENT = _static("parent.html")

    def _norm_story(self):
        start = self.PARENT.index("function normStory(")
        return self.PARENT[start:self.PARENT.index("async function getJSON", start)]

    def test_the_normaliser_keeps_who_told_it_and_why_it_failed(self):
        norm = self._norm_story()
        for field in ("storyteller:", "reason:", "parent_line:"):
            self.assertIn(field, norm)

    def test_the_page_says_who_told_each_story(self):
        self.assertIn("told by", self.PARENT)
        self.assertIn("s.storyteller", self.PARENT)

    def test_the_page_renders_the_sentence_a_grown_up_can_act_on(self):
        self.assertIn("s.parent_line", self.PARENT)
        # Escaped like every other string on this page: it is a model's words.
        self.assertIn("esc(s.parent_line", self.PARENT)


def tearDownModule():
    HTTPD.shutdown()
    LAN.shutdown()
    shutil.rmtree(HOME, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
