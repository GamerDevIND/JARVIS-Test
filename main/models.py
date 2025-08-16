import aiohttp
import json
import asyncio
import subprocess
import os # I hate my life; anyway this is for `env vars`
from utils import log

class Model:
    def __init__(self,role:str, name:str, ollama_name:str, has_tools:bool, has_CoT:bool, port:int, system_prompt:str) -> None:
        self.role = role
        self.name = name
        self.ollama_name = ollama_name
        self.has_tools = has_tools
        self.has_CoT = has_CoT
        self.port = port
        self.system = system_prompt

        self.host = f"http://localhost:{self.port}" # For seperate model loading

        self.start_command = ["ollama", "serve"]
        self.ollama_env = os.environ.copy()
        self.ollama_env["OLLAMA_HOST"] = self.host



        self.warmed_up = False

        self.session = None


    async def warm_up(self):
        self.process = subprocess.Popen(
                self.start_command, 
                env=self.ollama_env, 
                stdout=subprocess.DEVNULL, 
                # stderr=subprocess.STDOUT
        )


        print(f"🟨 [INFO] {self.name}({self.ollama_name}) warming up...")
        print("Trying Non-Streaming...")

        data = {
            "model": self.ollama_name,
            "messages": [{"role": "system", "content": self.system},{"role": "user", "content": "hi"}],
            "stream": False,
        }

        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.host}/api/chat" # For API calling
        headers = {"Content-Type": "application/json"}

        async with self.session.post(url, headers=headers, data=json.dumps(data)) as response:
            response.raise_for_status()

            response = await response.json()

            # print("✅ JSON response:", response)

            if 'message' in response and 'content' in response['message']:
                response['message']['content']
            else:
                print(f"🟥 [Error]: Unexpected API response format: {response}")
                return "An unexpected response format was received from the model."
        
        print('🟩 Non-Streaming works. Trying Streaming...')

        data['stream'] = True

        buffer = ""

        async with self.session.post(url, headers=headers, data=json.dumps(data)) as response:
                response.raise_for_status()

                async for chunk in response.content.iter_any():
                    buffer += chunk.decode("utf-8")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                # print(data["message"]["content"], end='', flush=True)
                                data["message"]["content"]
                        except json.JSONDecodeError:
                            continue
        self.warmed_up = True

        print(f"🟩 [INFO] {self.name}({self.ollama_name}) warmed up!")
        return ""

    async def generate_response_noStream(self, query:str, context = {}):
        url = f"{self.host}/api/chat" # For API calling

        messages = [{"role": "system", "content": self.system}] + context.get("conversations", []) + [{"role": "user", "content": query}]

        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.ollama_name,
            "messages": messages,
            "stream": False,
        }

        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            if not self.warmed_up:
                await self.warm_up()                
            else:
                print("Generating response...") # Normal generation
                async with self.session.post(url, headers=headers, data=json.dumps(data)) as response:
                        response.raise_for_status()

                        response = await response.json()

                        if 'message' in response and 'content' in response['message']:
                            return response['message']['content']
                        else:
                            print(f"🟥 [Error]: Unexpected API response format: {response}")
                            return "An unexpected response format was received from the model."
        except aiohttp.ClientError as e:
            print(f"🟥 [ERROR] Connection error: {e}")
            return f"Connection error: {e}"
        except json.JSONDecodeError:
            print(f"🟥 [Error] JSON decode error: Invalid JSON response.")
            return "Invalid JSON response from the model."
        except Exception as e:
            print(f"🟥 [Error]: {e}")
            return f"An unexpected error occurred: {e}"
        
    async def generate_response_Stream(self, query: str, context={}):
        url = f"{self.host}/api/chat"

        messages = [{"role": "system", "content": self.system}] + context.get("conversations", []) + [{"role": "user", "content": query}]

        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.ollama_name,
            "messages": messages,
            "stream": True,
        }

        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            if not self.warmed_up:
                await self.warm_up()

            async with self.session.post(url, headers=headers, data=json.dumps(data)) as response:
                response.raise_for_status()

                buffer = ""

                async for chunk in response.content.iter_any():
                    buffer += chunk.decode("utf-8")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                yield data["message"]["content"]
                        except json.JSONDecodeError:
                            continue

        except aiohttp.ClientError as e:
            print(f"🟥 [ERROR] Connection error: {e}")
            yield f"\n[Connection error: {e}]"
        except TimeoutError as e:
            print(f"🟥 Timeout Error: {e}")
            yield f"\n🟥 Timeout Error: {e}"
        except Exception as e:
            print(f"🟥 [ERROR] Unexpected: {e}")
            yield f"\n[Unexpected error: {e}]"


# ------- TEST -------

async def wait_until_ready(url: str, timeout: int = 20):
    print("Waiting for Ollama to be ready...")
    for i in range(timeout):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/api/tags") as res:
                    if res.status == 200:
                        print("🟩 Ollama is ready!")
                        return True
        except:
            print(f"Reties: {i+1} / {timeout}")
            pass
        await asyncio.sleep(1)
    raise TimeoutError(f"🟥 Ollama server did not start in time.")


async def main():
    prompts = [
        "Hi",
        "whats your name",
        "who made you?",
        "",
        "bye"
    ]

    m1 = Model("Testing", "Test", "llama3.2", True, False, 11434, "")
    m2 = Model("Testing2", "Test2", "llama3.2", True, False, 11435, "")
    ms = [m1, m2]

    c = {
        'conversations': []
    }

    for m in ms:
        subprocess.Popen(m.start_command, env=m.ollama_env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        await wait_until_ready(m.host)
        if not m.warmed_up:
            await m.generate_response_noStream("WARMING UP", c)


    for r in prompts:
        for m in ms:
            print(f"\n🟦 prompt: {r}\n⬛ response ({m.name}): ", end='', flush=True)

    
            stream = input('Stream? (y / n): ').lower() == 'y'

            if stream:
                collected = ""
                async for part in m.generate_response_Stream(r,c):
                    print(part, end='', flush=True)
                    collected += part
                
                c['conversations'].extend([
                    {"role": "user", "content": r},
                    {"role": "assistant", "content": collected}
                ])


    for m in ms:
        if m.session:
            await m.session.close()


if __name__ == "__main__":

    asyncio.run(main())