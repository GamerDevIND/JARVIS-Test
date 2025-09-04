import os
import json
import asyncio
import aiohttp
import subprocess
from utils import log

class Model:
    def __init__(self, role: str, name: str, ollama_name: str, has_tools: bool, has_CoT: bool, port: int, system_prompt: str) -> None:
        self.role = role
        self.name = name
        self.ollama_name = ollama_name
        self.has_tools = has_tools
        self.has_CoT = has_CoT
        self.port = port
        self.system = system_prompt
        self.host = f"http://localhost:{self.port}"
        self.start_command = ["ollama", "serve"]
        self.ollama_env = os.environ.copy()
        self.ollama_env["OLLAMA_HOST"] = self.host
        self.warmed_up = False
        self.session: aiohttp.ClientSession | None = None
        self.process = None

    async def wait_until_ready(self, url: str, timeout: int = 30):
        await log(f"Waiting for {self.name} on {url}...", "info")
        for i in range(timeout):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{url}/api/tags") as res:
                        if res.status == 200:
                            await log(f"{self.name} is ready!", "success")
                            return
            except aiohttp.ClientError:
                await log(f"Retries: {i+1} / {timeout}", "info")
            await asyncio.sleep(1)
        raise TimeoutError(f"🟥 Ollama server for {self.name} did not start in time.")

    async def warm_up(self):
        if self.warmed_up:
            return

        # Construct log file path using os.path.join
        log_dir = os.path.join("workspaces", "JARVIS-Test", "main", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"{self.ollama_name}.txt")
        
        await log(f"🟨 [INFO] {self.name} ({self.ollama_name}) warming up...", "info")
        with open(log_file_path, "w") as f:
            self.process = subprocess.Popen(
                self.start_command,
                env=self.ollama_env,
                stdout=f,
                stderr=subprocess.STDOUT
            )

        await self.wait_until_ready(self.host)
        
        if not self.session:
            self.session = aiohttp.ClientSession()

        # Perform a non-streaming test
        data = {
            "model": self.ollama_name,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }
        url = f"{self.host}/api/chat"
        headers = {"Content-Type": "application/json"}
        try:
            async with self.session.post(url, headers=headers, data=json.dumps(data)) as response:
                response.raise_for_status()
                res_json = await response.json()
                if not ('message' in res_json and 'content' in res_json['message']):
                    raise ValueError("Unexpected API response format during non-streaming test.")
        except (aiohttp.ClientError, ValueError) as e:
            await log(f"🟥 Non-streaming test failed for {self.name}: {e}", "error")
            return

        await log(f'🟩 Non-Streaming works. Trying Streaming...', "success")

        # Perform a streaming test
        data['stream'] = True
        try:
            async with self.session.post(url, headers=headers, data=json.dumps(data)) as response:
                response.raise_for_status()
                buffer = ""
                async for chunk in response.content.iter_any():
                    buffer += chunk.decode("utf-8")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line.strip():
                            try:
                                json_line = json.loads(line.strip())
                                # Assumes a streaming response contains a "message" object
                                if 'message' in json_line:
                                    break
                            except json.JSONDecodeError:
                                continue
        except (aiohttp.ClientError, json.JSONDecodeError) as e:
            await log(f"🟥 Streaming test failed for {self.name}: {e}", "error")
            return

        self.warmed_up = True
        await log(f"🟩 [INFO] {self.name} ({self.ollama_name}) warmed up!", "success")

    async def generate_response_noStream(self, query: str, context: dict) -> str:
        await log(f"Generating non-streaming response from {self.name}...", "info")
        url = f"{self.host}/api/chat"
        messages = context.get("conversations", []) + [{"role": "user", "content": query}]
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.ollama_name,
            "messages": messages,
            "stream": False,
        }
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.post(url, headers=headers, data=json.dumps(data)) as response:
                response.raise_for_status()
                res_json = await response.json()
                if 'message' in res_json and 'content' in res_json['message']:
                    return res_json['message']['content']
                else:
                    await log(f"🟥 [Error]: Unexpected API response format: {res_json}", "error")
                    return "An unexpected response format was received from the model."
        except aiohttp.ClientError as e:
            await log(f"🟥 [ERROR] Connection error: {e}", "error")
            return f"Connection error: {e}"
        except json.JSONDecodeError:
            await log(f"🟥 [Error] JSON decode error: Invalid JSON response.", "error")
            return "Invalid JSON response from the model."
        except Exception as e:
            await log(f"🟥 [Error]: {e}", "error")
            return f"An unexpected error occurred: {e}"

    async def generate_response_Stream(self, query: str, context: dict):
        await log(f"Generating streaming response from {self.name}...", "info")
        url = f"{self.host}/api/chat"
        messages = context.get("conversations", []) + [{"role": "user", "content": query}]
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.ollama_name,
            "messages": messages,
            "stream": True,
        }
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.post(url, headers=headers, data=json.dumps(data)) as response:
                response.raise_for_status()
                buffer = ""
                async for chunk in response.content.iter_any():
                    buffer += chunk.decode("utf-8")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line.strip():
                            try:
                                json_line = json.loads(line.strip())
                                if 'message' in json_line and 'content' in json_line['message']:
                                    yield json_line['message']['content']
                            except json.JSONDecodeError:
                                continue
        except aiohttp.ClientError as e:
            await log(f"🟥 [ERROR] Connection error: {e}", "error")
            yield f"\n[Connection error: {e}]"
        except TimeoutError as e:
            await log(f"🟥 Timeout Error: {e}", "error")
            yield f"\n🟥 Timeout Error: {e}"
        except Exception as e:
            await log(f"🟥 [ERROR] Unexpected: {e}", "error")
            yield f"\n[Unexpected error: {e}]"

    async def shutdown(self):
        await log(f"Shutting down {self.name}...", "info")
        if self.session is not None:
            await self.session.close()
            self.session = None
        if self.process is not None:
            self.process.terminate()
            await log(f"{self.name} process terminated.", "success")
            