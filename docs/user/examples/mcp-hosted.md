# Run a fatigue analysis in Codex or Claude Code

Connect to the hosted Py-Fatigue MCP, count cycles in a short stress
history, and ask a follow-up question about Miner damage.

You need Codex CLI or Claude Code and a bearer token from the Py-Fatigue
MCP operator. If you don't have a token, use the
[local MCP quickstart](../mcp.md#connect-codex).

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

## Count rainflow cycles

Send this prompt in either client.

```text
Use the py-fatigue-hosted MCP tool rainflow_count on these ordered
stress samples in MPa: [0, 75, -15, 50, 0].
Report each stress range and cycle weight, and the total cycle weight.
```

If the client prompts for approval, allow this `rainflow_count` call.
The tool returns stress ranges **75, 90, 65, and 50 MPa**. Each row
has weight **0.5**, for a total cycle weight of **2**.

## Ask a follow-up about Miner damage

```text
Use the py-fatigue-hosted MCP tool miner_damage on the original stress
samples. Use an illustrative S–N curve with slope 3 and log10 intercept
12. Report the total dimensionless Miner damage.
```

Approve the `miner_damage` call if prompted. The result is approximately
**`7.7525e-7`**. Check that the client called
`miner_damage` and used the original samples. The tool calculates the total
over all counted cycles.

If the server reports HTTP `401`, check that the token is valid and that
you launched your client from the shell where you exported it. To read
inputs from a file or export all result rows, use the
[local CSV example](mcp-fatigue.md).
