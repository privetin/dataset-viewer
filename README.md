# Dataset Viewer MCP Server
[![smithery badge](https://smithery.ai/badge/dataset-viewer)](https://smithery.ai/server/dataset-viewer)

An MCP server for interacting with the [Hugging Face Dataset Viewer API](https://huggingface.co/docs/dataset-viewer), providing capabilities to browse and analyze datasets hosted on the Hugging Face Hub.

## Features

### Resources

- Uses `dataset://` URI scheme for accessing Hugging Face datasets
- Supports dataset configurations and splits
- Provides paginated access to dataset contents
- Handles authentication for private datasets
- Supports searching and filtering dataset contents
- Provides dataset statistics and analysis

### Tools

The server provides the following tools:

1. **validate**
   - Check if a dataset exists and is accessible
   - Parameters:
     - `dataset`: Dataset identifier (e.g. 'stanfordnlp/imdb')
     - `auth_token` (optional): For private datasets

2. **get_info**
   - Get detailed information about a dataset
   - Parameters:
     - `dataset`: Dataset identifier
     - `auth_token` (optional): For private datasets

3. **get_rows**
   - Get paginated contents of a dataset
   - Parameters:
     - `dataset`: Dataset identifier
     - `config`: Configuration name
     - `split`: Split name
     - `page` (optional): Page number (0-based)
     - `auth_token` (optional): For private datasets

4. **get_first_rows**
   - Get first rows from a dataset split
   - Parameters:
     - `dataset`: Dataset identifier
     - `config`: Configuration name
     - `split`: Split name
     - `auth_token` (optional): For private datasets

5. **get_statistics**
   - Get statistics about a dataset split
   - Parameters:
     - `dataset`: Dataset identifier
     - `config`: Configuration name
     - `split`: Split name
     - `auth_token` (optional): For private datasets

6. **search_dataset**
   - Search for text within a dataset
   - Parameters:
     - `dataset`: Dataset identifier
     - `config`: Configuration name
     - `split`: Split name
     - `query`: Text to search for
     - `auth_token` (optional): For private datasets

7. **filter**
   - Filter rows using SQL-like conditions
   - Parameters:
     - `dataset`: Dataset identifier
     - `config`: Configuration name
     - `split`: Split name
     - `where`: SQL WHERE clause (e.g. "score > 0.5")
     - `orderby` (optional): SQL ORDER BY clause
     - `page` (optional): Page number (0-based)
     - `auth_token` (optional): For private datasets

8. **get_parquet**
   - Download entire dataset in Parquet format
   - Parameters:
     - `dataset`: Dataset identifier
     - `auth_token` (optional): For private datasets

## Installation

### Installing via Smithery

To install dataset-viewer for Claude Desktop automatically via [Smithery](https://smithery.ai/server/dataset-viewer):

```bash
npx -y @smithery/cli install dataset-viewer --client claude
```

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

### Setup

1. Clone the repository:
```bash
git clone https://github.com/privetin/dataset-viewer.git
cd dataset-viewer
```

2. Create a virtual environment and install:
```bash
# Create virtual environment
uv venv

# Activate virtual environment
# On Unix:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install in development mode
uv add -e .
```

## Configuration

### Environment Variables

- `HUGGINGFACE_TOKEN`: Your Hugging Face API token for accessing private datasets

### Claude Desktop Integration

Add the following to your Claude Desktop config file:

On Windows: `%APPDATA%\Claude\claude_desktop_config.json`

On MacOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "dataset-viewer": {
      "command": "uv",
      "args": [
        "run",
        "dataset-viewer"
      ]
    }
  }
}
```

## Usage Examples

1. Validate a dataset:
```json
{
  "dataset": "stanfordnlp/imdb"
}
```

2. Get dataset information:
```json
{
  "dataset": "stanfordnlp/imdb"
}
```

3. Search dataset contents:
```json
{
  "dataset": "stanfordnlp/imdb",
  "config": "plain_text",
  "split": "train",
  "query": "great movie"
}
```

4. Filter and sort rows:
```json
{
  "dataset": "stanfordnlp/imdb",
  "config": "plain_text",
  "split": "train",
  "where": "label = 'positive'",
  "orderby": "text DESC",
  "page": 0
}
```

5. Get dataset statistics:
```json
{
  "dataset": "stanfordnlp/imdb",
  "config": "plain_text",
  "split": "train"
}
```

## License

MIT License - see [LICENSE](LICENSE) for details