# Network Access Control in ragent

## The Key Insight: LLMs Cannot Make Network Calls

A model running in Ollama is a neural network — a function that takes tokens in and produces tokens out. It has no sockets, no HTTP client, no filesystem access. It cannot "browse the web" or "visit a URL" on its own.

When an agent "fetches a web page", what actually happens is:

```text
┌──────────────────────────────────────────────────────────┐
│  LLM (Ollama)        ragent (your code)        Internet  │
│                                                          │
│  outputs text:        parses tool call                   │
│  "call web_fetch  ─►  YOUR code makes  ─────────►   GET  │
│   (url='...')"        the HTTP request                   │
│                                                          │
│                       result fed back   ◄─────────  resp │
│  receives text   ◄──  to LLM as text                     │
└──────────────────────────────────────────────────────────┘
```

Every byte that leaves your machine on behalf of the agent goes through code **we** wrote. Ollama itself only listens on `localhost:11434` and does not initiate outbound connections during inference (it only does so when you `ollama pull` a model or check for updates).

The question "how do we stop the LLM from accessing unapproved sites" therefore reduces to: **how do we make sure the tool layer enforces the allowlist, and how do we verify it's working?**

---

## Layered Defense Strategy

Don't rely on a single layer. Each layer assumes the one above might fail.

```text
                     ┌─────────────────────────────────┐
  Layer 1            │  Tool layer (in ragent)         │  ← strongest, most semantic
  (in-process)       │  WebFetchTool with allowlist    │
                     └────────────────┬────────────────┘
                                      │ if a bug lets something through...
                     ┌────────────────▼────────────────┐
  Layer 2            │  ragent audit log (tracing)     │  ← evidence / forensics
  (in-process)       │  Every URL + result logged      │
                     └────────────────┬────────────────┘
                                      │
                     ┌────────────────▼────────────────┐
  Layer 3            │  OS-level firewall (per-process)│  ← belt-and-braces
  (out-of-process)   │  OpenSnitch / simplewall / etc. │
                     └────────────────┬────────────────┘
                                      │
                     ┌────────────────▼────────────────┐
  Layer 4            │  DNS / network monitor          │  ← passive observability
  (out-of-process)   │  Pi-hole / AdGuard Home         │
                     └─────────────────────────────────┘
```

---

### Layer 1: Tool Layer (Phase 5 of PLAN.md)

This is the most important layer and the only one we build ourselves. The `WebFetchTool` is the **only** place URLs become real network requests. If we get this right, the LLM can output `web_fetch(url="https://evil.com")` all day long and ragent will refuse.

**Requirements for a robust implementation:**

- **Default deny** — only whitelisted domains are allowed, configured per-project in `ragent.toml`
- **Parse and validate the URL** — extract the host ourselves, don't trust the LLM's claim
- **Block private/loopback IP ranges** — prevents SSRF (Server-Side Request Forgery) where an LLM tricks ragent into calling `http://192.168.1.1/admin` or `http://localhost:11434/...`
- **Strict host matching** — exact match + subdomain matching, NOT substring matching (`evil-docs.rs.attacker.com` must not match `docs.rs`)
- **Re-check after redirects** — don't let `reqwest` follow redirects to non-allowed domains blindly; re-validate the host at each hop
- **Cap response size and apply timeouts** — prevent the agent from downloading huge files or hanging on slow servers
- **Rate limiting** — prevent abuse even against allowed domains

**Example config (`ragent.toml`):**

```toml
[web]
allowed_domains = ["docs.rs", "doc.rust-lang.org", "crates.io"]
max_response_bytes = 1_048_576  # 1 MB
request_timeout_secs = 30
```

---

### Layer 2: Audit Logging (already partially in place)

The `tracing` infrastructure from Phase 1 already logs every HTTP call to Ollama. Extending it to `WebFetchTool` means every URL request, response status, and byte count goes into structured logs.

**Target structured event format:**

```rust
tracing::info!(
    target: "ragent::audit::web",
    url = %url,
    host = %host,
    allowed = allowed,
    status = response.status().as_u16(),
    bytes = body.len(),
    "web fetch"
);
```

A separate `tracing-subscriber` layer can write audit events to `audit.jsonl` for offline analysis. This is the forensic record: "what did the agent try to do?" — which matters even when the attempt was blocked.

---

### Layer 3: OS-Level Firewall (existing tools — don't build, install)

These tools hook into the OS network stack and see outbound SYN packets at the kernel level. No traffic interception, no MITM, no decryption — just allow/deny per process per destination.

| Tool | OS | Notes |
|------|----|----|
| **OpenSnitch** | Linux | Interactive per-app firewall, open-source |
| **Little Snitch** | macOS | Mature, paid |
| **LuLu** | macOS | Free alternative from Objective-See |
| **simplewall** | Windows | Open-source Windows Filtering Platform GUI |
| **Windows Firewall** | Windows | Built-in, can do per-app outbound rules manually |

**Recommended ragent rules:**

- Allow: `ragent → 127.0.0.1:11434` (Ollama)
- Allow: `ragent → <your allowed domains>:443`
- Default: deny everything else, with logging

---

### Layer 4: Passive Network Observability (existing tools)

For continuous monitoring without blocking decisions, run a logging DNS resolver.

| Tool | What it gives you |
|------|-------------------|
| **Pi-hole** | DNS server with web UI, logs every domain looked up |
| **AdGuard Home** | Similar, often easier to self-host |
| **dnsmasq + log-queries** | Minimal, just dumps queries to syslog |

For deeper visibility on Linux, eBPF tools from `bcc-tools` are lightweight and powerful:

- `tcpconnect -P 443` — prints every outbound TCP connection on port 443 with the originating PID
- `tcplife` — connection lifetime, bytes transferred, per-process
- Running `sudo tcpconnect-bpfcc -p $(pgrep ragent)` shows every connection ragent makes in real time

---

## What About a Separate "ragent-monitor" Service?

It could be a future learning project, but existing tools already handle connection-level enforcement better than anything we'd build.

Where a custom monitor **could** add value: a **semantic** monitor that understands tool calls and agent intent — tailing ragent's structured audit log and presenting "Agent X tried to fetch URL Y for project Z, denied because policy P". That's a dashboard over the audit log, not a network filter. This could be a Phase 7+ optional addition.

---

## Summary

| Concern | Answer |
|---------|--------|
| Can the LLM itself make network calls? | **No.** Only ragent's tool code can. |
| Where does network policy live? | **Layer 1: `WebFetchTool` allowlist** (Phase 5). |
| How do we audit what happened? | **Layer 2: structured `tracing` logs**, already half-built. |
| How do we enforce at the OS level? | **Layer 3: OpenSnitch / simplewall / etc.** Don't build, install. |
| How do we passively monitor? | **Layer 4: Pi-hole or DNS logger** for independent records. |
| Do we need a proxy? | **No.** We are the chokepoint (Layer 1). OS tools (Layer 3) work at the kernel level. |
| Do we need to build a separate service? | **Not for enforcement.** Possibly later for a semantic audit dashboard. |