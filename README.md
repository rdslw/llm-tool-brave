# llm-tool-brave

brave Search API plugin/tool for [Simon Willison's LLM](https://llm.datasette.io/).

Default api method: `context`, backed by brave's LLM Context API. 
It returns pre-extracted, token-budgeted grounding content for the model running in `llm`.
**NB:** This plugin does **not** shell out to `bx`;
it calls `https://api.search.brave.com/res/v1` directly with the Python standard library.

`AGENTS.md` points at this file, so keep operational project knowledge here.

## Demo

![Console demo of llm-tool-brave tool usage](demo.gif)

## Quick Start

### Quick in-place usage

```bash
uv sync                                     # see explanation below
uv run pytest                               # optional
uv run llm plugins                          # shall show llm-tool-brave
uv run llm tools list | grep -i brave
```

`uv sync` installs this package into `.venv` because `[tool.uv] package = true`.
The default `dev` dependency group makes `uv run pytest` work without extra flags.

Runtime-only local check:

```bash
uv run --no-dev llm plugins
uv run --no-dev llm tools list | grep -i brave
```

### Install into a global `llm` environment from local dir:

```bash
llm install -e .
llm plugins
```

### Install into a global `llm` after PyPi publishing:

```bash
llm install llm-tool-brave
```

## Configuration (one-time)

Set a brave Search API key:

```bash
llm keys set brave
```

Lookup order: stored key alias `brave`, `brave-search`, then env BRAVE_SEARCH_API_KEY.

Avoid raw API keys in toolbox constructor arguments such as
`-T 'Brave(api_key="...")'`. This is not the standard llm plugin approach
(key-requiring plugins resolve keys via `llm keys` or env vars), and llm 0.32+
persists toolbox constructor arguments to the logs database and replays them
when continuing conversations with `llm -c`. As a safety net this plugin scrubs
`api_key` from that persisted config, which means a conversation started with a
raw constructor key can only be continued if the key is also available via
`llm keys` or BRAVE_SEARCH_API_KEY. The `api_key` argument remains supported
for the Python API and tests.

## Usage

```bash
# default: only context tool is enabled; --td optional: shows tools calls by llm
llm -T Brave --td 'Search current web context for the latest llm tool plugin documentation and summarize it.'

# selected tools
llm -T 'Brave("context,web,news")' 'Find current uv release notes and summarize the important changes.'

# all implemented endpoints
llm -T 'Brave("all")' 'Find coffee shops near Warsaw city center and summarize the options.'
```

## Tools

| Tool | Endpoint | Plan | Use |
| --- | --- | --- | --- |
| `context` | `/llm/context` | Search | Main RAG/LLM grounding tool with extracted text chunks, tables, code, and source metadata. |
| `web` | `/web/search` | Search | Traditional web results with URLs, snippets, and rich result types. |
| `news` | `/news/search` | Search | Recent news with freshness filters. |
| `images` | `/images/search` | Search | Image results and thumbnails. |
| `videos` | `/videos/search` | Search | Video metadata. |
| `places` | `/local/place_search` | Search | Businesses, landmarks, POIs, cities, and addresses. |
| `suggest` | `/suggest/search` | Suggest | Query autocomplete and rich entity suggestions. |
| `spellcheck` | `/spellcheck/search` | Spellcheck | Pre-search query correction. |
| `answers` | `/chat/completions` | Answers | brave-generated answer. Prefer `context` when your current LLM should synthesize. |

Enable selected tools with `Brave("context,web,news")`; enable all with
`Brave("all")`.

Model-visible tool names carry the toolbox class prefix: `Brave_context`,
`Brave_web`, and so on. The prefix must match the registered class name so that
llm 0.32+ can map logged tool calls back to this toolbox and continue
conversations with `llm -c` without repeating the `-T` option.

Tool schemas are sent to the model with every request, so tool methods expose
only per-query parameters. Everything user- or environment-level lives on the
`Brave()` constructor instead (see Toolbox Settings) and never costs prompt
tokens.

## Context Tool

Use `Brave_context` first for web research, docs lookup, fact-checking, and RAG.

```bash
llm -T Brave --td 'Use brave context to find Python 3.13 pathlib changes and explain the practical impact.'
llm -T Brave --td 'Use brave context, restricted to docs.python.org and peps.python.org, to answer: what changed in Python 3.13 typing?'
```

Per-call parameters (the schema the model sees — deliberately short):

| Parameter | brave API field | Default | Range / values |
| --- | --- | --- | --- |
| `query` | `q` | required | search query |
| `count` | `count` | `20` | `1-50` |
| `max_tokens` | `maximum_number_of_tokens` | `8192` | `1024-32768` |
| `threshold` | `context_threshold_mode` | `balanced` | `strict`, `balanced`, `lenient`, `disabled` |
| `freshness` | `freshness` | none | `pd`, `pw`, `pm`, `py`, or `YYYY-MM-DDtoYYYY-MM-DD` |
| `include_sites`, `exclude_sites` | `goggles` | none | comma-separated domains |

Each call is bounded by `max_tokens` before returning to `llm`. Multiple searches
consume multiple tool responses plus JSON/source metadata. Invalid documented brave
ranges are rejected before the API request.

## Toolbox Settings

All other knobs are set once on the `Brave()` constructor and applied to every
call. With the llm CLI, pass either one positional value or keyword arguments —
the spec parser does not support mixing both:

```bash
llm -T 'Brave(tools="context,web", country="PL", search_lang="pl")' 'Find current uv release notes.'
```

| Setting | Default | Used by |
| --- | --- | --- |
| `country` | `US` | all tools |
| `search_lang` | `en` | all tools (also `lang` for suggest, spellcheck) |
| `safesearch` | `moderate` | web, images, videos |
| `spellcheck` | `true` | context, images |
| `goggles` | none | raw brave Goggles rules for context, web, news |
| `enable_local` | none | context |
| `extra_snippets` | none | web |
| `rich` | `false` | suggest |
| `units` | `metric` | places |
| `max_urls` | `20` (`1-50`) | context budget |
| `max_snippets` | `50` (`1-100`) | context budget |
| `max_tokens_per_url` | `4096` (`512-8192`) | context budget |
| `max_snippets_per_url` | `50` (`1-100`) | context budget |
| `lat`, `long`, `city`, `state`, `loc_country`, `postal_code` | none | context location headers |
| `research_iterations` | `3` | answers research mode |
| `research_seconds` | `120` | answers research mode |

Context budget settings are validated at construction time.

## Untrusted Web Content

Search result text is external and untrusted. The plugin wraps snippets,
descriptions, and brave-generated answer content with compact
`BRAVE_UNTRUSTED_CONTENT` markers and adds one `security_notice` field. URLs,
titles, source metadata, and structured fields stay intact.

This is a prompt-level guardrail, not a security boundary. Be careful combining
search tools with tools that can read private data, execute commands, send messages,
or exfiltrate information.

## Python API

```python
import llm
from llm_tool_brave import Brave

model = llm.get_model("gpt-5.4-mini")
response = model.chain(
    "Use brave context to find the current LLM tool plugin docs and summarize the API shape.",
    tools=[Brave()],
)
print(response.text())

tools = [Brave("context,web,news", country="PL", search_lang="pl")]
```

## Build And Release

```bash
uv build
```

Publishing uses `.github/workflows/publish.yml`: full Python test matrix, `uv build`,
then PyPI Trusted Publishing. Before the first release, configure PyPI for repository
`rdslw/llm-tool-brave`, workflow `publish.yml`, environment `pypi`.

`uv.lock` is intentionally not committed: this is a library, so the lockfile never
ships to users, and unlocked CI resolves fresh against `llm>=0.27` — an early
warning when a new `llm` release breaks the plugin. `[tool.uv] exclude-newer`
keeps brand-new package releases out of every resolution as a supply-chain guard.

Release:

1. Update `version` in `pyproject.toml`. PyPI versions are immutable.
2. Run `uv sync --all-groups`, `uv run pytest`, and `uv build`.
3. Commit, tag, push, then create a GitHub release from that tag.

```bash
git add pyproject.toml
git commit -m "Release 0.1.0"
git tag -a 0.1.0 -m "Release 0.1.0"
git push origin master
git push origin 0.1.0
```

The release event triggers `Publish to PyPI`. `workflow_dispatch` is available for
manual publishing, but normal versions should publish from release tags.

Post-publish smoke test:

```bash
uvx --from llm llm install llm-tool-brave
llm plugins
```

## Direct API Calls vs. `bx`

This plugin manually constructs HTTPS requests to the brave Search API instead of
invoking `bx`. Tradeoffs:

1. **Packaging:** direct API calls avoid requiring a separate `bx` binary.
2. **LLM ergonomics:** direct Python methods expose cleaner structured tool schemas.
3. **Secrets:** direct `llm keys` / `BRAVE_SEARCH_API_KEY` integration avoids command-line API keys.
4. **Feature coverage:** `bx` may expose brave API features, flags, config, and workflows not implemented here.
5. **Errors:** direct calls return structured tool errors; `bx` has mature CLI exit codes and diagnostics.
6. **Performance/control:** direct HTTP avoids subprocess and shell quoting overhead, but this plugin owns request/response behavior.
7. **API changes:** because `bx` is developed by brave, it is more likely to pick up brave Search API changes quickly.
8. **Shared utility:** an installed `bx` CLI is also useful for humans, scripts, and other tools, and is easy to test directly.
