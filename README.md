# Dataset Viewer MCP Server

An MCP server for interacting with the [Hugging Face Dataset Viewer API](https://huggingface.co/docs/dataset-viewer), providing capabilities to browse and analyze datasets hosted on the Hugging Face Hub.

## Features

### Resources

- Uses `dataset://` URI scheme for accessing Hugging Face datasets
- Supports dataset configurations and splits
- Provides paginated access to dataset contents

### Tools

The server provides three main tools:

1. **list_splits**
   - Lists available configurations and splits for a dataset
   - Required parameters:
     - `dataset_id`: Dataset identifier on Hugging Face Hub (e.g. 'username/dataset-name')

2. **view_dataset**
   - View paginated contents of a dataset
   - Required parameters:
     - `dataset_id`: Dataset identifier on Hugging Face Hub
     - `config`: Dataset configuration name (e.g. 'default')
     - `split`: Dataset split name (e.g. 'train', 'test')
   - Optional parameters:
     - `page`: Page number (0-based)

3. **get_stats**
   - Get statistics about a dataset
   - Required parameters:
     - `dataset_id`: Dataset identifier on Hugging Face Hub
     - `config`: Dataset configuration name
     - `split`: Dataset split name

## Installation

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

1. List available splits for a dataset:
```json
{
  "dataset_id": "cornell-movie-review-data/rotten_tomatoes"
}
```

2. View dataset contents:
```json
{
  "dataset_id": "cornell-movie-review-data/rotten_tomatoes",
  "config": "default",
  "split": "train",
  "page": 0
}
```

3. Get dataset statistics:
```json
{
  "dataset_id": "cornell-movie-review-data/rotten_tomatoes",
  "config": "default",
  "split": "train"
}
```

## Limitations

- Only works with datasets hosted on the Hugging Face Hub
- Maximum page size of 100 rows when viewing dataset contents
- Requires dataset name to be in format 'username/dataset-name'

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add appropriate license]