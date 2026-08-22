# Story Lantern

A bedside lamp your kid talks to.

They ask for a story. Twelve seconds later a painted page appears and a warm voice starts
reading a brand-new story out loud — and their characters come back next time looking and
acting the same. The dog Biscuit still has the folded ear and the red collar.

Everything happens on a **Tiiny Pocket** on your own network. The story is written,
illustrated and narrated by a box on the nightstand. No account, no subscription, no cloud
API, and nothing about your child leaves the house.

---

## What it actually does

| | |
|---|---|
| **Writes** the story | Ornith-1.0-35B, page by page, ~60 words a page |
| **Paints** every page | Z-Image-Turbo, 512×512, 8 diffusion steps, ~8s |
| **Reads** it aloud | Qwen3-TTS CustomVoice — a warm, unhurried voice |
| **Remembers** characters | a character bible: a fixed descriptive phrase plus a fixed image seed, replayed verbatim into every later illustration |
| **Checks** every page | a safety layer that classifies each page *before* it can be shown or spoken |
| **Shows parents** everything | a transparency log: what was asked, what was written, and the verdict on every page |

The character memory is the product. A five-year-old notices when the dog looks right.

---

## Requirements

- A **Tiiny Pocket** reachable on your network
- Any machine to host the lantern: a Raspberry Pi, a mini PC, an old laptop
- Python 3.11+ — **standard library only**, no pip install, no virtualenv, no build step
- A browser for the display (kiosk mode on a small screen makes it a lamp)

NPU budget: Ornith 50u + Z-Image 32u + TTS 7u. TTS is loaded for a session and released
after, so it co-exists with other things using the device.

## Run it in ten minutes

```bash
git clone https://github.com/webdevtodayjason/story-lantern
cd story-lantern

export TIINY_HOST=192.168.1.50          # your device
export TIINY_KEY=<your device api key>  # Settings → API in TiinyOS

python3 lantern.py --selfcheck          # one short story end to end, prints timings
python3 lantern.py                      # serve it
```

Then open **http://localhost:8420/show** for the lantern and
**http://localhost:8420/parent** for the transparency log.

For an actual bedside lamp, point a small screen at it in kiosk mode:

```bash
chromium-browser --kiosk --autoplay-policy=no-user-gesture-required \
  --app=http://127.0.0.1:8420/show
```

The autoplay flag is not optional. Without it the narration silently never plays.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `TIINY_HOST` | — | device address (required) |
| `TIINY_KEY` | — | device API key (required) |
| `PORT` | `8420` | web port |
| `LANTERN_DB` | `./lantern.db` | SQLite state |
| `LANTERN_BLOCKLIST` | `./blocklist_extra.txt` | extra words for the deterministic backstop |
| `LANTERN_ALLOW_UNVERIFIED` | *unset* | **leave it unset.** See below. |

---

## About the safety layer

An unmoderated model generating bedtime content for a five-year-old is the failure mode
that matters, so this is the part that got the most attention.

Three things happen before a child hears a word:

1. **A deterministic backstop** runs on the request before any model is involved. A
   catastrophic phrase never depends on a model being in a good mood.
2. **A charter check** on what the child asked for. If it needs declining, the lantern
   redirects warmly — it never shows a child an error or a refusal.
3. **A classifier pass on every generated page**, against a rubric written for ages 3–8:
   no violence beyond fairy-tale peril, no death of a named companion, no body horror, no
   adult themes, nothing that frightens a child at bedtime, nothing instructing real-world
   danger. A failed page is regenerated once with a gentler steer; if it fails again, a
   pre-written safe page is substituted and the substitution is logged loudly.

**It fails closed.** If the device cannot be reached to classify a page, the page is *not*
shown. `LANTERN_ALLOW_UNVERIFIED=1` takes the other side of that trade knowingly; the
default does not. This was a deliberate reversal during review — the earlier behaviour
showed unverified pages and merely badged them, which is the wrong default when the
audience is asleep in ten minutes.

Every verdict is persisted. The parent log hides nothing, including the pages that were
blocked and why.

This layer is not a guarantee. It is a serious, auditable effort, and you should read
`safety.py` yourself before pointing this at your own child. `python3 safety.py` runs an
adversarial self-test against a table of both innocent and genuinely nasty requests.

---

## Design notes worth knowing

**The device does one inference at a time.** A second concurrent call fails with error
`150004`. Every device call in this project goes through a single `DeviceWorker` thread
and retries `150004` with backoff, so Story Lantern co-exists with other applications
using the same Tiiny rather than fighting them.

**Prefetch is what makes it feel instant.** While page 1's narration plays (~30 seconds),
the device is already writing and painting page 2 (~8 seconds). After the first page there
is no waiting, ever — no spinner, no gap, just a cross-fade.

**512×512 is not an aesthetic choice.** On current firmware it is the only image size
Z-Image-Turbo will render; every other size fails after ~30 seconds.

**The chat model puts its reasoning in `reasoning_content`, and it counts against
`max_tokens`.** Ask for JSON with a small budget and you get an empty response. This
project uses a generous budget and scavenges the last complete JSON object out of the
reasoning text if `content` comes back empty.

**A child never sees an error.** If an illustration fails, the prose appears alone on a
warm plate. If narration fails, the words stay on screen to be read. There are no
spinners, no percentages and no stack traces anywhere in the display.

---

## Making it yours

The obvious extensions, roughly in order of how much fun they are:

- **Voice input.** ASR (`Qwen3-ASR-1.7B`, 7u) and a `getUserMedia` capture in the display,
  so the child speaks instead of typing. Left out of this version only because the NPU
  budget was shared with another application during development.
- **Your own art style.** The illustration prompt prefix is one constant. Swap gouache for
  crayon, woodcut, cut-paper.
- **A physical button.** A cheap arcade button that enumerates as a keyboard sending
  `Space` needs no GPIO code at all.
- **More voices.** A different narrator per character.
- **Other languages.** The story prompt and the safety rubric are the only places language
  is assumed.

---

## Credits

Built on the [Tiiny Pocket](https://tiiny.ai/) by Tiiny AI, whose PowerInfer and
TurboSparse work is the reason a device this size can run a 35B model at all.

Built by [Jason Brashear](https://github.com/webdevtodayjason).

Python standard library only. One HTML file per view. No dependencies, no build step,
no telemetry, no network egress beyond your own device.
