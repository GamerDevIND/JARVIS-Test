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
        warmup_tasks = [model.warm_up() for model in self.models.values()]
        await asyncio.gather(*warmup_tasks)

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

    async def route_query(self, query: str, manual: str | None = None) -> str:
        if manual is not None:
            if manual == "think":
                return "cot"
            elif manual == "chat":
                self.models["chat"].system = CHAT_PROMPT
                return "chat"
            elif manual == "chaos":
                self.models["chat"].system = CHAOS_PROMPT
                return "chat"

        router = self.models.get("router")
        if not router:
            await log("Router model not found, using default model.", "error")
            return self.default_model

        try:
            response = await router.generate_response_noStream(query, self.context)
            if response and response.strip() in self.models:
                await log(f"Router selected model: {response.strip()}", "info")
                return response.strip()
            else:
                await log(f"Router returned unknown model: {response}, using default.", "error")
                return self.default_model
        except Exception as e:
            await log(f"Error during routing: {e}, using default.", "error")
            return self.default_model

    async def generate(self, query: str):
        if not query:
            return
            
        if query.startswith("!"):
            parts = query.split(" ", 1)
            command = parts[0][1:]
            query = parts[1] if len(parts) > 1 else ""
            model_name = await self.route_query(query, manual=command)
        else:
            model_name = await self.route_query(query)

        model = self.models.get(model_name)
        if not model:
            yield "Sorry, the requested model is not available."
            return

        stream = self.platform.lower() not in STREAM_DISABLED
        await log(f"Using stream: {stream}", "info")
        poll = model.process.poll() # type: ignore

        if poll is not None:
            await log(f"Restarting model {model.name}", "warn")
            await model.warm_up()

        full_response = ""
        if stream:
            async for part in model.generate_response_Stream(query, self.context):
                full_response += part
                yield part
        else:
            response_text = await model.generate_response_noStream(query, self.context)
            full_response = response_text
            yield response_text

        self.context['conversations'].append({"role": "user", "content": query})
        self.context['conversations'].append({"role": "assistant", "content": full_response})
        await self.save_context()

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
            async for part in ai.generate(req):
                print(part, end="", flush=True)
            print()
        except Exception as e:
            await log(f"Main loop error: {e}", "error")
            break

if __name__ == "__main__":
    asyncio.run(main())