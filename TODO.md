## List of TODO tasks for llm-tool-brave

* Analyze and enhance tool snippets/description
  - is tools.description good for 'brave-context' ? check `tmp/conversation_dump.md` and observe that '### Tools' section (subsection of '## Prompt') has quite long arguments json part. Is it really worthwile to have it full at every prompt/llm call provided with our tool???
  - do we need to provide ALL properties in input schema? it uses a lot of tokens

* Based on llm-tool-brave create llm-tool-bx which will simply use 'bx' CLI tool instead of http api
  - plan to be defined...

* Stage two untrusted-content hardening:
  - add broader recursive coverage for all brave response fields that can contain external page text
  - consider a fuller warning mode for fetched/full-page content if that endpoint is added later
  - detect/log suspicious prompt-injection phrases for debugging without blocking legitimate results
  - harden marker sanitization against homoglyph and zero-width spoofing patterns, following the fuller openclaw-style approach
  - add compatibility tests using captured brave response fixtures for context, web, news, images, videos, and places
