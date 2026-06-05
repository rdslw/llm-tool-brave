## List of TODO tasks for llm-tool-brave

* Based on llm-tool-brave create llm-tool-bx which will simply use 'bx' CLI tool instead of http api
  - plan to be defined...

* Stage two untrusted-content hardening:
  - add broader recursive coverage for all brave response fields that can contain external page text
  - consider a fuller warning mode for fetched/full-page content if that endpoint is added later
  - detect/log suspicious prompt-injection phrases for debugging without blocking legitimate results
  - harden marker sanitization against homoglyph and zero-width spoofing patterns, following the fuller openclaw-style approach
  - add compatibility tests using captured brave response fixtures for context, web, news, images, videos, and places
