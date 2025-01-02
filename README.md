# dataset-viewer MCP server

MCP server for interacting with Hugging Face dataset viewer API, providing dataset browsing, filtering, and statistics capabilities

## Components

### Resources

The server implements a simple note storage system with:
- Custom note:// URI scheme for accessing individual notes
- Each note resource has a name, description and text/plain mimetype

### Prompts

The server provides a single prompt:
- summarize-notes: Creates summaries of all stored notes
  - Optional "style" argument to control detail level (brief/detailed)
  - Generates prompt combining all current notes with style preference

### Tools

The server implements one tool:
- add-note: Adds a new note to the server
  - Takes "name" and "content" as required string arguments
  - Updates server state and notifies clients of resource changes

## Configuration

[TODO: Add configuration details specific to your implementation]

## Quickstart

### Install

#### Claude Desktop

On MacOS: `~/Library/Application\ Support/Claude/claude_desktop_config.json`

On Windows: `C:\Users\<username>\AppData\Roaming\Claude\claude_desktop_config.json`

<details>
  <summary>Development/Unpublished Servers Configuration</summary>

  ```
  "mcpServers": {
    "dataset-viewer": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\MCP\server\community\dataset-viewer",
        "run",
        "dataset-viewer"
      ]
    }
  }
  ```
</details>

<details>
  <summary>Published Servers Configuration</summary>

  ```
  "mcpServers": {
    "dataset-viewer": {
      "command": "uvx",
      "args": [
        "dataset-viewer"
      ]
    }
  }
  ```
</details>

## Development

### Building and Publishing

To prepare the package for distribution:

1. Sync dependencies and update lockfile:
```bash
uv sync
```

2. Build package distributions:
```bash
uv build
```

This will create source and wheel distributions in the `dist/` directory.

3. Publish to PyPI:
```bash
uv publish
```

Note: You'll need to set PyPI credentials via environment variables or command flags:
- Token: `--token` or `UV_PUBLISH_TOKEN`
- Or username/password: `--username`/`UV_PUBLISH_USERNAME` and `--password`/`UV_PUBLISH_PASSWORD`

### Debugging

Since MCP servers run over stdio, debugging can be challenging. For the best debugging
experience, we strongly recommend using the [MCP Inspector](https://github.com/modelcontextprotocol/inspector).


You can launch the MCP Inspector via [`npm`](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) with this command:

```bash
npx @modelcontextprotocol/inspector uv --directory C:\MCP\server\community\dataset-viewer run dataset-viewer
```


Upon launching, the Inspector will display a URL that you can access in your browser to begin debugging.