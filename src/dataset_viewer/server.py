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

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl, BaseModel
import mcp.server.stdio


class DatasetViewerAPI:
    """Internal API client for dataset viewer"""
    def __init__(self, base_url: str = "https://datasets-server.huggingface.co"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(base_url=self.base_url)
        
    def _validate_dataset_id(self, dataset_id: str) -> None:
        """Validate dataset ID format (username/dataset-name)"""
        if not dataset_id or "/" not in dataset_id:
            raise ValueError(
                "Invalid dataset ID format. Must be 'username/dataset-name', e.g. 'cornell-movie-review-data/rotten_tomatoes'"
            )
        
    async def get_dataset_info(self, dataset_id: str) -> dict:
        """Get basic information about a dataset"""
        self._validate_dataset_id(dataset_id)
        
        # First check if dataset is valid
        response = await self.client.get(f"/is-valid", params={"dataset": dataset_id})
        response.raise_for_status()
        
        # Get splits info
        response = await self.client.get(f"/splits", params={"dataset": dataset_id})
        response.raise_for_status()
        return response.json()
        
    async def get_dataset_content(self, dataset_id: str, config: str, split: str, page: int = 0) -> dict:
        """Get paginated content of a dataset"""
        self._validate_dataset_id(dataset_id)
        
        params = {
            "dataset": dataset_id,
            "config": config,
            "split": split,
            "offset": page * 100,  # 100 rows per page
            "length": 100
        }
            
        response = await self.client.get(f"/rows", params=params)
        response.raise_for_status()
        return response.json()

    async def get_dataset_stats(self, dataset_id: str, config: str, split: str) -> dict:
        """Get statistics about a dataset"""
        self._validate_dataset_id(dataset_id)
        
        params = {
            "dataset": dataset_id,
            "config": config,
            "split": split
        }
            
        response = await self.client.get(f"/statistics", params=params)
        response.raise_for_status()
        return response.json()


class DatasetState:
    """Manages dataset state and caching"""
    def __init__(self):
        self.datasets: dict[str, dict] = {}  # Cache dataset info
        self.current_page: dict[str, int] = {}  # Track pagination
        self.api = DatasetViewerAPI()

    async def get_dataset(self, dataset_id: str) -> dict:
        """Get dataset info, using cache if available"""
        if dataset_id not in self.datasets:
            self.datasets[dataset_id] = await self.api.get_dataset_info(dataset_id)
        return self.datasets[dataset_id]


# Initialize server and state
server = Server("dataset-viewer")
state = DatasetState()


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available dataset resources"""
    resources = []
    for dataset_id, info in state.datasets.items():
        resources.append(
            types.Resource(
                uri=AnyUrl(f"dataset://{dataset_id}"),
                name=dataset_id,
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

    dataset_id = uri.path
    if dataset_id is not None:
        dataset_id = dataset_id.lstrip("/")
        info = await state.get_dataset(dataset_id)
        return str(info)  # Convert to string for display
    raise ValueError(f"Dataset not found: {dataset_id}")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available dataset tools for Hugging Face datasets"""
    return [
        types.Tool(
            name="list_splits",
            description="List available configurations and splits for a Hugging Face dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset identifier on Hugging Face Hub (e.g. 'username/dataset-name')"},
                },
                "required": ["dataset_id"],
            },
        ),
        types.Tool(
            name="view_dataset",
            description="View contents of a Hugging Face dataset with pagination",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset identifier on Hugging Face Hub (e.g. 'username/dataset-name')"},
                    "config": {"type": "string", "description": "Dataset configuration name (e.g. 'default')"},
                    "split": {"type": "string", "description": "Dataset split name (e.g. 'train', 'test')"},
                    "page": {"type": "integer", "description": "Page number (0-based)"},
                },
                "required": ["dataset_id", "config", "split"],
            },
        ),
        types.Tool(
            name="get_stats",
            description="Get statistics for a Hugging Face dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset identifier on Hugging Face Hub (e.g. 'username/dataset-name')"},
                    "config": {"type": "string", "description": "Dataset configuration name (e.g. 'default')"},
                    "split": {"type": "string", "description": "Dataset split name (e.g. 'train', 'test')"},
                },
                "required": ["dataset_id", "config", "split"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution requests"""
    if arguments is None:
        arguments = {}

    if name == "list_splits":
        dataset_id = arguments["dataset_id"]
        info = await state.api.get_dataset_info(dataset_id)
        return [
            types.TextContent(
                type="text",
                text=f"Available splits for dataset '{dataset_id}':\n{str(info)}"
            )
        ]

    elif name == "view_dataset":
        dataset_id = arguments["dataset_id"]
        config = arguments["config"]
        split = arguments["split"]
        page = arguments.get("page", 0)
        
        content = await state.api.get_dataset_content(
            dataset_id, config=config, split=split, page=page
        )
        state.current_page[dataset_id] = page
        
        # Notify about state change
        await server.request_context.session.send_resource_list_changed()
        
        return [
            types.TextContent(
                type="text",
                text=str(content)
            )
        ]

    elif name == "get_stats":
        dataset_id = arguments["dataset_id"]
        config = arguments["config"]
        split = arguments["split"]
        stats = await state.api.get_dataset_stats(dataset_id, config=config, split=split)
        return [
            types.TextContent(
                type="text",
                text=str(stats)
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
                    name="dataset_id",
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

    if not arguments or "dataset_id" not in arguments:
        raise ValueError("Missing dataset_id argument")

    dataset_id = arguments["dataset_id"]
    info = await state.get_dataset(dataset_id)
    
    return types.GetPromptResult(
        description=f"Analyze dataset: {dataset_id}",
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