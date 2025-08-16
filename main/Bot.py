import discord
import os
from AI import AI
import asyncio

DISCORD_KEY = os.getenv("DISCORD_KEY", "")

intents = discord.Intents.default()
intents.message_content = True

ai = AI()

class Bot(discord.Client):
    async def on_ready(self):
        print(f'Logged in as {self.user}')

        await ai.init("discord",True)

        print("Ready!")

    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return
        
        # print(f"{message.author.display_name } ({message.author}): {message.content}")
        think = None

        att = message.attachments
        add = ""
        if att:
            for a in att:
                    print(a.filename)
                    if a.filename.split(".")[-1] in ['py', 'txt']: # type: ignore
                        print("file detected")
                        c = await a.read()
                        decoded = c.decode(errors="ignore")
                        add = f"(user sent a file which reads:\n{decoded})"


        query:str = (add + "\n" + message.content).strip()
        if not query:
            return
        await message.channel.typing()
        try:
            response = None
            async for part in ai.generate(query,):
                response = part
            try:
                if response is not None:
                    think, response = response.split('</think>') # type:ignore
                    think = think.replace("<think>", '').replace("</think>", "")
            except:
                pass
            await message.channel.typing()
        except Exception as e:
            print(f"[Error] {e}")
            response = f"Oops! Something went wrong. Try again later. (`{e}`)"

        print(think is not None and think.strip())

        if response is not None:
            await message.reply(response)


        if think is not None and think.strip():
            await message.channel.send(f'Thinking:\n```\n{think}\n```')
        else: # Debuger
            await message.channel.send("-# NO THINKING")
        # print("AI: " + response)

async def start_discord_bot():
    """Starts and manages the Discord bot's lifecycle."""
    bot = Bot(intents=intents)
    try:
        if not DISCORD_KEY:
            raise ValueError("DISCORD_KEY is missing.")
        await bot.start(DISCORD_KEY)
    except Exception as e:
        print(f"🟥 An unhandled error occurred in start_discord_bot: {e}")
    finally:
        if bot.is_ready():
            await bot.close() 
        await ai.shut_down()
        print("PULSE AI system shut down...")

if __name__ == "__main__":
    asyncio.run(start_discord_bot()) 