#!/usr/bin/env python3
"""MCP server: detect faces/heads in a still image using Party-Pose's YOLOX model."""

import sys
sys.path.insert(0, "/home/thesa/Projects/Party-Pose")

import asyncio
import json
import cv2
from pathlib import Path

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from src.model_manager import ModelManager
from src.person_detector import PersonDetector

MODELS_DIR = Path("/home/thesa/Projects/Party-Pose/models")

_detector: PersonDetector | None = None

def get_detector() -> PersonDetector:
    global _detector
    if _detector is None:
        mm = ModelManager(MODELS_DIR)
        _detector = PersonDetector(mm)
    return _detector

server = Server("party-pose-face-detector")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="detect_faces",
            description=(
                "Detect heads/faces in a still image using the Party-Pose YOLOX model. "
                "Draws green bounding boxes and saves an annotated copy. "
                "Returns output_path and a list of {x,y,w,h,confidence} boxes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute path to the input image (jpg/png/webp/etc.)",
                    }
                },
                "required": ["image_path"],
            },
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name != "detect_faces":
        raise ValueError(f"Unknown tool: {name}")

    image_path = Path(arguments.get("image_path", ""))
    if not image_path.exists():
        return [types.TextContent(type="text", text=json.dumps({"error": f"File not found: {image_path}"}))]

    try:
        frame_bgr = cv2.imread(str(image_path))
        if frame_bgr is None:
            return [types.TextContent(type="text", text=json.dumps({"error": "Could not decode image"}))]

        h, w = frame_bgr.shape[:2]

        frame_gpu = cv2.cuda_GpuMat()
        frame_gpu.upload(frame_bgr)

        detections = get_detector().process(frame_gpu, w, h)

        faces = []
        for box in detections.head_boxes:
            faces.append({
                "x": float(box.x), "y": float(box.y),
                "w": float(box.w), "h": float(box.h),
                "confidence": float(box.confidence),
            })
            cv2.rectangle(
                frame_bgr,
                (int(box.x), int(box.y)),
                (int(box.x + box.w), int(box.y + box.h)),
                (0, 255, 0), 2,
            )
            cv2.putText(
                frame_bgr, f"{box.confidence:.2f}",
                (int(box.x), int(box.y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        output_path = image_path.with_name(f"{image_path.stem}_faces{image_path.suffix}")
        cv2.imwrite(str(output_path), frame_bgr)

        return [types.TextContent(type="text", text=json.dumps({
            "output_path": str(output_path),
            "face_count": len(faces),
            "faces": faces,
        }))]

    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream,
            InitializationOptions(
                server_name="party-pose-face-detector",
                server_version="1.0.0",
                capabilities=server.get_capabilities(NotificationOptions(), {}),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
