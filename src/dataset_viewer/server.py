"""MCP Server for interacting with Hugging Face dataset viewer API.

This server provides tools for browsing, filtering and getting statistics about datasets hosted on the 
Hugging Face Hub. It uses the official dataset viewer API (https://huggingface.co/docs/dataset-viewer)
to provide:

- Dataset validation and basic info
- Paginated content viewing 
- Dataset statistics
- Support for dataset configurations and splits

Note: This only works with datasets hosted on the Hugging Face Hub. For local datasets or datasets from
other sources, you'll need to upload them to Hugging Face first.
"""

import asyncio
from typing import Optional
import httpx
import os
import re
import json

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl, BaseModel
import mcp.server.stdio


class DatasetViewerAPI:
    """Internal API client for dataset viewer"""
    def __init__(self, base_url: str = "https://datasets-server.huggingface.co", auth_token: str | None = None):
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
        self.client = httpx.AsyncClient(base_url=self.base_url, headers=headers)

    async def validate_dataset(self, dataset: str) -> None:
        """Validate dataset ID format and check if it exists"""
        # Validate format (username/dataset-name)
        if not re.match(r"^[^/]+/[^/]+$", dataset):
            raise ValueError("Dataset ID must be in the format 'owner/dataset'")
            
        # Check if dataset exists and is accessible
        try:
            response = await self.client.head(f"/is-valid?dataset={dataset}")
            response.raise_for_status()
        except httpx.NetworkError as e:
            raise ConnectionError(f"Network error while validating dataset: {e}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Dataset '{dataset}' not found")
            elif e.response.status_code == 403:
                raise ValueError(f"Dataset '{dataset}' exists but requires authentication")
            else:
                raise RuntimeError(f"Error validating dataset: {e}")

    async def get_info(self, dataset: str) -> dict:
        """Get detailed information about a dataset"""
        try:
            # Get detailed dataset info
            response = await self.client.get("/info", params={"dataset": dataset})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Dataset '{dataset}' not found")
            raise
            
    async def get_rows(self, dataset: str, config: str, split: str, page: int = 0) -> dict:
        """Get paginated rows of a dataset"""
        params = {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": page * 100,  # 100 rows per page
            "length": 100
        }
        response = await self.client.get("/rows", params=params)
        response.raise_for_status()
        return response.json()

    async def get_statistics(self, dataset: str, config: str, split: str) -> dict:
        """Get statistics about a dataset"""
        params = {
            "dataset": dataset,
            "config": config,
            "split": split
        }
        response = await self.client.get("/statistics", params=params)
        response.raise_for_status()
        return response.json()
        
    async def get_first_rows(self, dataset: str, config: str, split: str) -> dict:
        """Get first few rows of a dataset split"""
        params = {
            "dataset": dataset,
            "config": config,
            "split": split
        }
        response = await self.client.get("/first-rows", params=params)
        response.raise_for_status()
        return response.json()
        
    async def search(self, dataset: str, config: str, split: str, query: str) -> dict:
        """Search for text within a dataset split"""
        params = {
            "dataset": dataset,
            "config": config,
            "split": split,
            "query": query
        }
        response = await self.client.get("/search", params=params)
        response.raise_for_status()
        return response.json()

    async def filter(self, dataset: str, config: str, split: str, where: str, orderby: str | None = None, page: int = 0) -> dict:
        """Filter dataset rows based on conditions"""
        # Validate page number
        if page < 0:
            raise ValueError("Page number must be non-negative")
            
        # Basic SQL clause validation
        if not where.strip():
            raise ValueError("WHERE clause cannot be empty")
        if orderby and not orderby.strip():
            raise ValueError("ORDER BY clause cannot be empty")
            
        params = {
            "dataset": dataset,
            "config": config,
            "split": split,
            "where": where,
            "offset": page * 100,  # 100 rows per page
            "length": 100
        }
        if orderby:
            params["orderby"] = orderby
            
        try:
            response = await self.client.get("/filter", params=params)
            response.raise_for_status()
            return response.json()
        except httpx.NetworkError as e:
            raise ConnectionError(f"Network error while filtering dataset: {e}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                raise ValueError(f"Invalid filter query: {e.response.text}")
            elif e.response.status_code == 404:
                raise ValueError(f"Dataset, config or split not found: {dataset}/{config}/{split}")
            else:
                raise RuntimeError(f"Error filtering dataset: {e}")

    async def get_parquet(self, dataset: str) -> bytes:
        """Get entire dataset in Parquet format"""
        response = await self.client.get("/parquet", params={"dataset": dataset})
        response.raise_for_status()
        return response.content

    async def get_splits(self, dataset: str) -> dict:
        """Get list of available splits for a dataset"""
        response = await self.client.get("/splits", params={"dataset": dataset})
        response.raise_for_status()
        return response.json()


class DatasetState:
    """Manages dataset state and caching"""
    def __init__(self):
        self.datasets: dict[str, dict] = {}  # Cache dataset info
        self.current_page: dict[str, int] = {}  # Track pagination
        # Get auth token from environment if available
        auth_token = os.environ.get("HUGGINGFACE_TOKEN")
        self.api = DatasetViewerAPI(auth_token=auth_token)

    async def get_dataset(self, dataset: str) -> dict:
        """Get dataset info, using cache if available"""
        if dataset not in self.datasets:
            self.datasets[dataset] = await self.api.get_info(dataset)
        return self.datasets[dataset]


# Initialize server and state
server = Server("dataset-viewer")
state = DatasetState()


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available dataset resources"""
    resources = []
    for dataset, info in state.datasets.items():
        resources.append(
            types.Resource(
                uri=AnyUrl(f"dataset://{dataset}"),
                name=dataset,
                description=info.get("description", "No description available"),
                mimeType="application/json",
            )
        )
    return resources


@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """Read a specific dataset's content"""
    if uri.scheme != "dataset":
        raise ValueError(f"Unsupported URI scheme: {uri.scheme}")

    dataset = uri.path
    if dataset is not None:
        dataset = dataset.lstrip("/")
        info = await state.get_dataset(dataset)
        return str(info)  # Convert to string for display
    raise ValueError(f"Dataset not found: {dataset}")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available dataset tools for Hugging Face datasets"""
    return [
        types.Tool(
            name="get_info",
            description="Get detailed information about a Hugging Face dataset including description, features, splits, and statistics. Run validate first to check if the dataset exists and is accessible.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Hugging Face dataset identifier in the format owner/dataset",
                        "pattern": "^[^/]+/[^/]+$",
                        "examples": ["ylecun/mnist", "stanfordnlp/imdb"]
                    },
                    "auth_token": {
                        "type": "string",
                        "description": "Hugging Face auth token for private/gated datasets",
                        "optional": True
                    }
                },
                "required": ["dataset"],
            }
        ),
        types.Tool(
            name="get_rows",
            description="Get paginated rows from a Hugging Face dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Hugging Face dataset identifier in the format owner/dataset",
                        "pattern": "^[^/]+/[^/]+$",
                        "examples": ["ylecun/mnist", "stanfordnlp/imdb"]
                    },
                    "config": {
                        "type": "string",
                        "description": "Dataset configuration/subset name. Use get_info to list available configs",
                        "examples": ["default", "en", "es"]
                    },
                    "split": {
                        "type": "string",
                        "description": "Dataset split name. Splits partition the data for training/evaluation",
                        "examples": ["train", "validation", "test"]
                    },
                    "page": {"type": "integer", "description": "Page number (0-based), returns 100 rows per page", "default": 0},
                    "auth_token": {
                        "type": "string",
                        "description": "Hugging Face auth token for private/gated datasets",
                        "optional": True
                    }
                },
                "required": ["dataset", "config", "split"],
            }
        ),
        types.Tool(
            name="get_first_rows",
            description="Get first rows from a Hugging Face dataset split",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Hugging Face dataset identifier in the format owner/dataset",
                        "pattern": "^[^/]+/[^/]+$",
                        "examples": ["ylecun/mnist", "stanfordnlp/imdb"]
                    },
                    "config": {
                        "type": "string",
                        "description": "Dataset configuration/subset name. Use get_info to list available configs",
                        "examples": ["default", "en", "es"]
                    },
                    "split": {
                        "type": "string",
                        "description": "Dataset split name. Splits partition the data for training/evaluation",
                        "examples": ["train", "validation", "test"]
                    },
                    "auth_token": {
                        "type": "string",
                        "description": "Hugging Face auth token for private/gated datasets",
                        "optional": True
                    }
                },
                "required": ["dataset", "config", "split"],
            }
        ),
        types.Tool(
            name="search_dataset",
            description="Search for text within a Hugging Face dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Hugging Face dataset identifier in the format owner/dataset",
                        "pattern": "^[^/]+/[^/]+$",
                        "examples": ["ylecun/mnist", "stanfordnlp/imdb"]
                    },
                    "config": {
                        "type": "string",
                        "description": "Dataset configuration/subset name. Use get_info to list available configs",
                        "examples": ["default", "en", "es"]
                    },
                    "split": {
                        "type": "string",
                        "description": "Dataset split name. Splits partition the data for training/evaluation",
                        "examples": ["train", "validation", "test"]
                    },
                    "query": {"type": "string", "description": "Text to search for in the dataset"},
                    "auth_token": {
                        "type": "string",
                        "description": "Hugging Face auth token for private/gated datasets",
                        "optional": True
                    }
                },
                "required": ["dataset", "config", "split", "query"],
            }
        ),
        types.Tool(
            name="filter",
            description="Filter rows in a Hugging Face dataset using SQL-like conditions",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Hugging Face dataset identifier in the format owner/dataset",
                        "pattern": "^[^/]+/[^/]+$",
                        "examples": ["ylecun/mnist", "stanfordnlp/imdb"]
                    },
                    "config": {
                        "type": "string",
                        "description": "Dataset configuration/subset name. Use get_info to list available configs",
                        "examples": ["default", "en", "es"]
                    },
                    "split": {
                        "type": "string",
                        "description": "Dataset split name. Splits partition the data for training/evaluation",
                        "examples": ["train", "validation", "test"]
                    },
                    "where": {
                        "type": "string",
                        "description": "SQL-like WHERE clause to filter rows",
                        "examples": ["column = \"value\"", "score > 0.5", "text LIKE \"%query%\""]
                    },
                    "orderby": {
                        "type": "string",
                        "description": "SQL-like ORDER BY clause to sort results",
                        "optional": True,
                        "examples": ["column ASC", "score DESC", "name ASC, id DESC"]
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number for paginated results (100 rows per page)",
                        "default": 0,
                        "minimum": 0
                    },
                    "auth_token": {
                        "type": "string",
                        "description": "Hugging Face auth token for private/gated datasets",
                        "optional": True
                    }
                },
                "required": ["dataset", "config", "split", "where"],
            }
        ),
        types.Tool(
            name="get_statistics",
            description="Get statistics about a Hugging Face dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Hugging Face dataset identifier in the format owner/dataset",
                        "pattern": "^[^/]+/[^/]+$",
                        "examples": ["ylecun/mnist", "stanfordnlp/imdb"]
                    },
                    "config": {
                        "type": "string",
                        "description": "Dataset configuration/subset name. Use get_info to list available configs",
                        "examples": ["default", "en", "es"]
                    },
                    "split": {
                        "type": "string",
                        "description": "Dataset split name. Splits partition the data for training/evaluation",
                        "examples": ["train", "validation", "test"]
                    },
                    "auth_token": {
                        "type": "string",
                        "description": "Hugging Face auth token for private/gated datasets",
                        "optional": True
                    }
                },
                "required": ["dataset", "config", "split"],
            }
        ),
        types.Tool(
            name="get_parquet",
            description="Export Hugging Face dataset split as Parquet file",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Hugging Face dataset identifier in the format owner/dataset",
                        "pattern": "^[^/]+/[^/]+$",
                        "examples": ["ylecun/mnist", "stanfordnlp/imdb"]
                    },
                    "auth_token": {
                        "type": "string",
                        "description": "Hugging Face auth token for private/gated datasets",
                        "optional": True
                    }
                },
                "required": ["dataset"],
            }
        ),
        types.Tool(
            name="validate",
            description="Check if a Hugging Face dataset exists and is accessible",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string", 
                        "description": "Hugging Face dataset identifier in the format owner/dataset",
                        "pattern": "^[^/]+/[^/]+$",
                        "examples": ["ylecun/mnist", "stanfordnlp/imdb"]
                    },
                    "auth_token": {
                        "type": "string",
                        "description": "Hugging Face auth token for private/gated datasets",
                        "optional": True
                    }
                },
                "required": ["dataset"],
            }
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution requests"""
    if arguments is None:
        arguments = {}

    # Allow overriding env token with explicit token
    auth_token = arguments.pop("auth_token", None) or os.environ.get("HUGGINGFACE_TOKEN")

    if name == "get_info":
        dataset = arguments["dataset"]
        try:
            response = await DatasetViewerAPI(auth_token=auth_token).client.get("/info", params={"dataset": dataset})
            response.raise_for_status()
            result = response.json()
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )
            ]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Dataset '{dataset}' not found"
                    )
                ]
            raise

    elif name == "get_rows":
        dataset = arguments["dataset"]
        config = arguments["config"]
        split = arguments["split"]
        page = arguments.get("page", 0)
        rows = await DatasetViewerAPI(auth_token=auth_token).get_rows(dataset, config=config, split=split, page=page)
        return [
            types.TextContent(
                type="text",
                text=json.dumps(rows, indent=2)
            )
        ]

    elif name == "get_first_rows":
        dataset = arguments["dataset"]
        config = arguments["config"]
        split = arguments["split"]
        first_rows = await DatasetViewerAPI(auth_token=auth_token).get_first_rows(dataset, config=config, split=split)
        return [
            types.TextContent(
                type="text",
                text=json.dumps(first_rows, indent=2)
            )
        ]

    elif name == "search_dataset":
        dataset = arguments["dataset"]
        config = arguments["config"]
        split = arguments["split"]
        query = arguments["query"]
        search_result = await DatasetViewerAPI(auth_token=auth_token).search(dataset, config=config, split=split, query=query)
        return [
            types.TextContent(
                type="text",
                text=json.dumps(search_result, indent=2)
            )
        ]

    elif name == "filter":
        dataset = arguments["dataset"]
        config = arguments["config"]
        split = arguments["split"]
        where = arguments["where"]
        orderby = arguments.get("orderby")
        page = arguments.get("page", 0)
        filtered = await DatasetViewerAPI(auth_token=auth_token).filter(dataset, config=config, split=split, where=where, orderby=orderby, page=page)
        return [
            types.TextContent(
                type="text",
                text=json.dumps(filtered, indent=2)
            )
        ]

    elif name == "get_statistics":
        dataset = arguments["dataset"]
        config = arguments["config"]
        split = arguments["split"]
        stats = await DatasetViewerAPI(auth_token=auth_token).get_statistics(dataset, config=config, split=split)
        return [
            types.TextContent(
                type="text",
                text=json.dumps(stats, indent=2)
            )
        ]

    elif name == "get_parquet":
        dataset = arguments["dataset"]
        parquet_data = await DatasetViewerAPI(auth_token=auth_token).get_parquet(dataset)
        
        # Save to a temporary file with .parquet extension
        filename = f"{dataset.replace('/', '_')}.parquet"
        filepath = os.path.join(os.getcwd(), filename)
        with open(filepath, "wb") as f:
            f.write(parquet_data)
            
        return [
            types.TextContent(
                type="text",
                text=f"Dataset exported to: {filepath}"
            )
        ]

    elif name == "validate":
        dataset = arguments["dataset"]
        try:
            # First check format
            if not re.match(r"^[^/]+/[^/]+$", dataset):
                return [
                    types.TextContent(
                        type="text",
                        text="Dataset must be in the format 'owner/dataset'"
                    )
                ]
                
            # Then check if dataset exists and is accessible
            response = await DatasetViewerAPI(auth_token=auth_token).client.get("/is-valid", params={"dataset": dataset})
            response.raise_for_status()
            result = response.json()
            
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )
            ]
        except httpx.NetworkError as e:
            return [
                types.TextContent(
                    type="text",
                    text=str(e)
                )
            ]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Dataset '{dataset}' not found"
                    )
                ]
            elif e.response.status_code == 403:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Dataset '{dataset}' requires authentication"
                    )
                ]
            else:
                return [
                    types.TextContent(
                        type="text",
                        text=str(e)
                    )
                ]
    raise ValueError(f"Unknown tool: {name}")


@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """List available prompts for dataset analysis"""
    return [
        types.Prompt(
            name="analyze-dataset",
            description="Analyze a dataset's content and structure",
            arguments=[
                types.PromptArgument(
                    name="dataset",
                    description="Dataset identifier",
                    required=True,
                )
            ],
        )
    ]

@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """Generate dataset analysis prompts"""
    if name != "analyze-dataset":
        raise ValueError(f"Unknown prompt: {name}")

    if not arguments or "dataset" not in arguments:
        raise ValueError("Missing dataset argument")

    dataset = arguments["dataset"]
    info = await state.get_dataset(dataset)
    
    return types.GetPromptResult(
        description=f"Analyze dataset: {dataset}",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Please analyze this dataset:\n\n{str(info)}",
                ),
            )
        ],
    )


async def main():
    """Run the server using stdin/stdout streams"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="dataset-viewer",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())