# 3. Connect to hosted MCP

Use an MCP client with Streamable HTTP and bearer-token support.
Get a bearer token from the Py-Fatigue MCP operator. You can also
[run the MCP locally](02-run-locally.md).

| Connection field | Value |
| --- | --- |
| URL | `https://mcp.py-fatigue.dev/mcp` |
| Transport | Streamable HTTP |
| Authorization header | `Bearer <your token>` |

Enter the token using your client’s credential settings if available.
The two CLI configurations below read it from an environment variable.

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

## Verify the connection

List the server’s tools in your client. You should see
`rainflow_count`, `miner_damage`, and `ordered_crack_growth`. If you
get HTTP `401`, check your token and how the client loads it.

If your server lists `lffd_recovery`, you can also
[recover cycles across stress windows](06-recover-lffd.md).

Continue with the [inline stress history example](04-analyze-stress-history.md).
