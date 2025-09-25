import asyncio
import aiofiles
import json
from utils import log
from models import Model
from configs import ( 
    CoT_PROMPT, 
    CHAT_PROMPT, 
    ROUTER_PROMPT, 
    DEFAULT_PROMPT, 
    CHAOS_PROMPT, 
    STREAM_DISABLED 
)

class AI:
    def __init__(self, model_config_path="main/Models_config.json", context_path="main/saves/context.json"):
        self.model_config_path = model_config_path
        self.context_path = context_path
        self.models: dict[str, Model] = {}
        self.system_prompts = {
            "chat": CHAT_PROMPT,
            "router": ROUTER_PROMPT,
            "cot": CoT_PROMPT,
        }
        self.default_model = 'chat'
        self.load_models()

    def load_models(self):
        try:
            with open(self.model_config_path, 'r', encoding="utf-8") as f:
                models_data = json.load(f)
                for model_data in models_data:
                    role = model_data.get('role')
                    if role:
                        model_data["system_prompt"] = self.system_prompts.get(role, DEFAULT_PROMPT)
                        self.models[role] = Model(**model_data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"🟥 Error loading models: {e}")
            exit(1)

    async def init(self, platform: str):
        self.context = await self.load_context()
        self.platform = platform
        await log("Warming up all models...", "info")
        for model in self.models.values():
            await model.warm_up()
            await asyncio.sleep(0.02) 

    async def load_context(self):
        try:
            async with aiofiles.open(self.context_path) as file:
                content = await file.read()
                return json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"conversations": []}

    async def save_context(self):
        try:
            async with aiofiles.open(self.context_path, "w") as file:
                await file.write(json.dumps(self.context, indent=2))
        except IOError as e:
            await log(f"🟥 Error saving context: {e}", "error")

    async def check_models(self, interval=10):
        while True:
            for role, model in self.models.items():
                if model.process and model.process.poll() is not None:
                    await log(f"⚠️ {model.name} crashed. Restarting...", "warn")
                    await model.warm_up()
            await asyncio.sleep(interval)


    async def shut_down(self):
        await log("Shutting Down all services...", "info")
        shutdown_tasks = [model.shutdown() for model in self.models.values()]
        await asyncio.gather(*shutdown_tasks)
        await self.save_context()
        print("Done.")

async def main():
    ai = AI()
    await ai.init("cli")
    while True:
        req = input(">>> ")
        if req == "/bye":
            await ai.shut_down()
            break
        
        try:
            # async for part in ai.generate(req):
            #     print(part, end="", flush=True)
            print()
        except Exception as e:
            await log(f"Main loop error: {e}", "error")
            break

if __name__ == "__main__":
    asyncio.run(main())