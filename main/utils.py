import requests
from datetime import datetime
import asyncio
import aiofiles
import json
from spin import Spinner

async def log(message: str, level, log_file: str = "/workspaces/JARVIS-Test/main/logs/log.log", append = True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emoji = {
        "info": "ℹ️",
        "warning": "⚠️",
        "error": "🟥",
        "success": "✅"
    }.get(level, "")
    text = f"{emoji} [{level}] {message} - [{timestamp}]\n"
    async with aiofiles.open(log_file, "a" if append else "w") as f:
        print(text)
        await f.write(text)