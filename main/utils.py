import requests
from datetime import datetime
import asyncio
import aiofiles
import json
from spin import Spinner

async def log(message: str, level, log_file: str = "main/logs/log.txt", append = True):
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


async def main():
    s = Spinner("⚙️  Spinning up Pulse AI...")
    spinner_task = asyncio.create_task(s.spinny_thingy())

    from models import main as pulse_main

    await pulse_main()
    s.change_text("NEW")
    await pulse_main()

    s.set_done()
    await spinner_task
    print("✅ Spinny done!\n")

if __name__ == "__main__":
    asyncio.run(main())
