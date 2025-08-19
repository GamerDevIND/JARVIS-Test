from configs import ( CoT_PROMPT, 
                     CHAT_PROMPT, 
                     ROUTER_PROMPT, 
                     DEFAULT_PROMPT, 
                     CHAOS_PROMPT, 
                     STREAM_DISABLED )

import asyncio
import aiofiles
from models import Model
import json
from utils import log
import subprocess

# Sooo ... now what?

default_model = 'chat'  

class AI:
    def __init__(self, model_config_path= "main/Models_config.json",context_path="main/saves/context.json", memory_path = "main/saves/memory.json") -> None:
        self.model_config_path = model_config_path
        self.context_path = context_path
        self.memory_path = memory_path
        self.models :dict[str,Model] = dict()

        self.tools = [
             None
        ]

        system_prompts= {
            "chat": CHAT_PROMPT,
            'router': ROUTER_PROMPT,
            "cot": CoT_PROMPT,
        }

        models = self.load_models()
        # print(models)
        if models is not None:
            for model in models:
                model["system_prompt"] = system_prompts.get(model['role'], DEFAULT_PROMPT)
                self.models[model["role"]] = Model(**model)
        else:
            print("🟥 NO MODELS FOUND exiting...")
            exit(1)

    async def init(self, platform:str, auto_warmup = False, ):
        self.context = await self.load_context()
        self.platform:str = platform
        if auto_warmup:
            async for _ in self.generate("", False):
                continue

    async def warm_up(self, model:Model):
            await model.generate_response_noStream("")

    async def route(self, query:str, manual = False):
        if manual:
            if query.startswith("!think"):
                return "cot"
            elif query.startswith("!chat"):
                self.models["chat"].system = CHAT_PROMPT
                return "chat"
            elif query.startswith("!chaos"):
                self.models["chat"].system = CHAOS_PROMPT
                print("chaos")
                return "chat"
            else:
                return default_model
            
        else:
            router = self.models.get("router")
            if router is None:
                await log("Router model not found, using default model.", "error")
                return default_model
            await log(f"Routing query: {query}", "info")
            try:
                response = await router.generate_response_noStream(query, self.context)
                if response:
                    response = response.strip()
                    # response= json.loads(response)
                    await log(f"Router response: {response}", "info")
                    if response in self.models.keys():
                        return response
                    else:
                        await log(f"Router returned an unknown model: {response}, using default model.", "error")
                        return default_model
            except Exception as e:
                await log(f"Error during routing: {e}", "error")
                return default_model

    async def generate(self, query:str, save = True):
        for model in self.models.values():
            if not model.warmed_up: 
                await self.warm_up(model)

        # Normal normal generation
        if query == "":
            return
        response = await self.route(query)
        print("router: " , response)
        if response:
            model_name = response 
        else:
            model_name = default_model
        if model_name:
            model = self.models[model_name.lower()]
            print(model.name)
            print(query)

            stream = not (self.platform.lower() in STREAM_DISABLED)
            print(f"Using steam: {stream}")
            if not stream:
                response = await model.generate_response_noStream(query, self.context)
                if response:
                    response = response
                if save:
                    self.context['conversations'].extend([{"role":"user", "content": query}, {"role":"assistant", "content": response}]) # type: ignore
                    
                    await self.save_context()

                yield response

            else:
                part = ""
                response = part
                async for part in model.generate_response_Stream(query,self.context):
                    if part == "":
                        break
                    else:
                        response += part
                    yield part

                if save:       
                    self.context['conversations'].extend([
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": response}
                    ])
                    await self.save_context()

    async def shut_down(self):
        print("Shutting Down...")
        for model in self.models.values():
            if model.session is not None:
                await model.session.close()
            if model.process is not None:
                model.process.terminate()
            
        subprocess.Popen(["ollama", "stop", model.ollama_name])
        await self.save_context()
        print("Done.")

    async def load_context(self): 
        try:
            async with aiofiles.open(self.context_path) as file:
                content = await file.read()
                return json.loads(content)
        except FileNotFoundError:
            return {"conversations": []}
        except json.JSONDecodeError as e:
            await log(f"JSON Error: {e}", "error")
            return {"conversations": []}


    async def save_context(self):
        async with aiofiles.open(self.context_path, "w") as file:
            await file.write(json.dumps(self.context, indent=2))

    def load_models(self) : 
        try:
            with open(self.model_config_path, 'r', encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"🟥 File Error: {e}")
        except Exception as e:
            print(f"An error occured: {e}")

        return None

async def main():
    ai = AI()
    await ai.init("cli", True) 
    while True:
        req = input(">>> ")
        if req == "/bye":
            await ai.shut_down()
            break
        async for part in ai.generate(req,):
            print(part, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())  
