# 3. Connect to hosted MCP

Use an MCP client with Streamable HTTP and bearer-token support.
You can also [run MCP locally without a token](02-run-locally.md).

## Request a token

Email [pietro.dantuono@vub.be](mailto:pietro.dantuono@vub.be) to request
hosted access. Include your name, your organization if applicable, and
how you plan to use the service. The operator will share the token
privately if access is approved. Use the value you receive as
`PY_FATIGUE_MCP_TOKEN` in your client.

| Connection field | Value |
| --- | --- |
| URL | `https://mcp.py-fatigue.dev/mcp` |
| Transport | Streamable HTTP |
| Authorization header | `Bearer <your token>` |

Enter the token using your client’s credential settings if available.
The CLI examples below read it from an environment variable.

## Load your token

In Bash, enter the token at the hidden prompt. Launch your client from
this shell so it can read the exported variable.

```bash
read -rsp 'Py-Fatigue MCP token: ' PY_FATIGUE_MCP_TOKEN
printf '\n'
export PY_FATIGUE_MCP_TOKEN
```

## Connect your client

### Codex CLI

Register the server once. The configuration stores the environment
variable name, and Codex reads its value when the session starts.

```sh
codex mcp add py-fatigue-hosted \
  --url https://mcp.py-fatigue.dev/mcp \
  --bearer-token-env-var PY_FATIGUE_MCP_TOKEN
codex
```

In the new session, run `/mcp` and look for `py-fatigue-hosted`.
[Codex MCP configuration](https://developers.openai.com/codex/mcp)

### Claude Code

In the directory where you plan to run Claude Code, create a
`.mcp.json` file with this configuration. If you already have one,
add the `py-fatigue-hosted` entry to its `mcpServers` object.

```json
{
  "mcpServers": {
    "py-fatigue-hosted": {
      "type": "http",
      "url": "https://mcp.py-fatigue.dev/mcp",
      "headers": {
        "Authorization": "Bearer ${PY_FATIGUE_MCP_TOKEN}"
      }
    }
  }
}
```

Start `claude` from that directory. Approve the project MCP configuration
if prompted, then run `/mcp` and look for `py-fatigue-hosted`.
[Claude Code MCP configuration](https://code.claude.com/docs/en/mcp#environment-variable-expansion-in-mcpjson)

### Zed

Open **Settings → AI → MCP Servers** and choose
**Add Server → Add Remote Server**. Name it `py-fatigue-hosted` and use
`https://mcp.py-fatigue.dev/mcp` as the URL.

To configure the header in JSON, run `zed: open settings file` from the
Command Palette and merge the entry into your personal `context_servers`
settings. Replace `<your token>` with the privately supplied value and
keep the token out of shared project settings and Git.

```json
{
  "context_servers": {
    "py-fatigue-hosted": {
      "url": "https://mcp.py-fatigue.dev/mcp",
      "headers": {
        "Authorization": "Bearer <your token>"
      }
    }
  }
}
```

Check for the active indicator beside `py-fatigue-hosted` in
**Settings → AI → MCP Servers**. Open the Agent Panel and mention the
server by name when you try the
[inline stress-history example](04-analyze-stress-history.md).
[Zed MCP setup](https://zed.dev/docs/ai/mcp#as-custom-servers)

## Verify the connection

List the server’s tools in your client. You should see
`rainflow_count`, `miner_damage`, and `ordered_crack_growth`. If you
get HTTP `401`, check your token and how the client loads it.

Tool availability follows the server version. Use the
[local 0.2.0 setup](02-run-locally.md) if the hosted instance
does not advertise the newer tools. Hosted requests use inline samples
or synthetic specifications. PNG and ECharts results are returned to
the client without writing files on the host.

Continue with the [inline stress history example](04-analyze-stress-history.md).
