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

## Operator instructions

### Grant access

The HTTP server accepts one shared token from `PY_FATIGUE_MCP_BEARER_TOKEN`.
Retrieve the current value from the deployment's secret configuration and
share it through a private channel, such as a password manager's sharing
feature. Send the recipient this guide so they can set
`PY_FATIGUE_MCP_TOKEN` and verify their connection. Sharing the current
token requires no restart.

### Rotate the token

Replacing the token invalidates the old value for every existing client.
Generate a replacement in a private terminal with `openssl rand -hex 32`.
Set the replacement as `PY_FATIGUE_MCP_BEARER_TOKEN` in the deployment's
existing secret configuration. Keep the value out of Git and shell command
arguments. The server reads it at startup, so recreate the container to
apply the updated environment.

On the Docker host, list the running containers and set `container` to
the MCP container name. Check the Compose labels to identify how it was
started.

```bash
sudo docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}'
container='your-mcp-container-name'
sudo docker inspect --format \
  'project={{index .Config.Labels "com.docker.compose.project"}} service={{index .Config.Labels "com.docker.compose.service"}}' \
  "$container"
```

**Docker Compose.** If the Compose labels are populated, run the command
from the deployment directory. Use the existing Compose file, project
and environment options, and replace `your-mcp-service` with the service
from the label.

```bash
sudo docker compose up -d --no-deps --force-recreate your-mcp-service
```

**Standalone Docker.** Have the original launch command ready before
stopping the container. Keep the stopped container available for rollback.

```bash
sudo docker stop "$container"
sudo docker rename "$container" "${container}-before-rotation"
```

Rerun the original `docker run` command with the same image, container
name, ports, mounts, and other options, using the updated secret
configuration. If verification fails and restoring the old token is
appropriate, remove the replacement container if it exists, then restore
and start the original.

```bash
sudo docker rm -f "$container"
sudo docker rename "${container}-before-rotation" "$container"
sudo docker start "$container"
```

### Verify the rotation

Load the replacement token in a fresh client session and confirm that it
can list the hosted tools. Check that the old token receives HTTP `401`.
In Bash, enter the old token at the hidden prompt.

```bash
read -rsp 'Previous Py-Fatigue MCP token: ' previous_token
printf '\n'
printf 'Authorization: Bearer %s\n' "$previous_token" |
  curl --silent --show-error --output /dev/null --write-out '%{http_code}\n' \
    --header @- https://mcp.py-fatigue.dev/mcp
unset previous_token
```

Send the replacement privately to authorized users so they can update
`PY_FATIGUE_MCP_TOKEN`. Once the standalone container is verified and
rollback is no longer needed, remove the stopped container.

```bash
sudo docker rm "${container}-before-rotation"
```
