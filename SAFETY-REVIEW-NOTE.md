# RESOLVED — classifier-unreachable now fails closed

**Raised:** during the overnight build, against `safety.py`'s `ALLOW_UNVERIFIED` path.
**Closed:** 2026-08-22 by the integration pass. Nothing outstanding.

## What was wrong

If the independent classifier could not be reached inside its retry budget — a plausible
event, because a WARBOARD instance shares this device and returns 150004 under contention
— the page was shown and spoken anyway, recorded as `ALLOW_UNVERIFIED` and badged amber in
the parent log.

The rationale was not silly: the deterministic blocklist and the request-level charter had
both already passed, and refusing every page during a contention window would end a child's
story for a reason unrelated to content. But the default was still wrong for this product.
The audience is a five-year-old at bedtime, and "we logged it loudly" is no help in the
moment, because the child has already heard it. The cost of failing closed is near zero,
because the warm pre-written closing page already exists for the `BLOCK` case.

## What changed

`StorySession._judge` in `lantern.py`:

- Classifier unreachable now returns `UNVERIFIED`, which splices the bundled closing page.
  The child gets a slightly shorter story; they never get an unverified one. It gets its
  own page verdict (`UNVERIFIED_FALLBACK`) and its own paragraph in the parent log,
  because "two drafts failed" and "nothing could be checked" are different facts and the
  log used to report both as the first.
- A `CLASSIFIER_UNAVAILABLE` row still goes into `safety_event` with the page text verbatim,
  so the parent log shows exactly what was held back and why.
- Fail-open is still available, but it is now an explicit, knowing opt-in:
  `LANTERN_ALLOW_UNVERIFIED=1`. On that path the page is badged `ALLOW_UNVERIFIED` in the
  parent log, which already had the amber styling for it.

- A verdict the classifier returns that is NOT one of ALLOW/SOFTEN/BLOCK — a missing
  `verdict` key, or a synonym like UNSAFE or REJECT — used to default to `ALLOW`, which
  was the exact inverse of the policy on this page and of what `safety.py` had always
  done. It is now `SOFTEN`: one regeneration, then the fallback page.
- The page text is now delimited when it is handed to the classifier, with a standing
  instruction that nothing inside the markers is an instruction, and the classifier's
  reply is parsed from `message.content` only. Scavenging a safety verdict out of
  `reasoning_content` meant a JSON object the model quoted back while thinking could
  become the verdict.

Two related gaps were closed in the same pass:

- The deterministic backstop's `BLOCK` was being downgraded to `SOFTEN` for everything,
  which meant a page that produced gore got a polite "try again" instead of ending the
  story. Catastrophic categories (`gore`, `sexual`, `self_harm`, `real_violence`, `drugs`,
  `hate`, `abuse_disclosure`, `danger_howto`, `injection`, `body_horror`) now go straight to
  the fallback page. Regeneration is for a story that leaned tense, not for one that
  produced something that should never be regenerated toward.
- `safety.backstop_request`'s `story_steer` was computed and then dropped on the floor. It
  is now carried into both the plan prompt and every page prompt.
