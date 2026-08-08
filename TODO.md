## List of TODO tasks for llm-tool-brave

* llm 0.32 follow-ups. 0.10.0 made the plugin 0.32-compatible (Brave_ tool
  prefix for llm -c continuation, api_key scrubbed from persisted toolbox
  config) while keeping llm>=0.27. Three optional 0.32-era features remain;
  the first two would force bumping to llm>=0.32:
  - PauseChain (llm>=0.32): raise llm.PauseChain from expensive/slow tools for
    human approval before the request is sent - e.g. answers() with
    enable_research=True burns a pricier Answers-plan query and can run for two
    minutes. Could be a constructor flag like Brave(confirm="answers,places").
  - llm_tool_call parameter (llm>=0.32): reserved tool-method parameter carrying
    the unique tool_call_id, never shown to the model. Use it as the
    BRAVE_UNTRUSTED_CONTENT marker id and as an error-dict correlation tag; see
    the TODO(llm>=0.32) comments in llm_tool_brave.py.
  - prepare() lifecycle hook (llm>=0.31) - analyze first whether it is worth it:
    could fail fast on a missing API key before the model burns a turn, but it
    runs even for llm tools listings, which currently work without any key.

* Based on llm-tool-brave create llm-tool-bx which will simply use 'bx' CLI tool instead of http api
  - plan to be defined...

* Stage two untrusted-content hardening:
  - add broader recursive coverage for all brave response fields that can contain external page text
  - consider a fuller warning mode for fetched/full-page content if that endpoint is added later
  - detect/log suspicious prompt-injection phrases for debugging without blocking legitimate results
  - harden marker sanitization against homoglyph and zero-width spoofing patterns, following the fuller openclaw-style approach
  - add compatibility tests using captured brave response fixtures for context, web, news, images, videos, and places
