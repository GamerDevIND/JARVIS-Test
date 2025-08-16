import sys
import asyncio
import colorama
from itertools import cycle

class Spinner:
    def __init__(self, text) -> None:
        self.done = asyncio.Event()
        self.text = text
        self.chars = ["-", "\\", "|", '/']
        colorama.init(True)
        sys.stdout.write(colorama.Fore.RESET + colorama.Back.RESET + "\r")

    async def spinny_thingy(self, color = colorama.Fore.LIGHTCYAN_EX):
        sys.stdout.write(" " * len(self.text) + '\r')
        while not self.done.is_set():
            for i in cycle(self.chars):
                if self.done.is_set():
                    break

                sys.stdout.write(f"{color}{self.text} {i}\r{colorama.Fore.RESET}")
                sys.stdout.flush()
                await asyncio.sleep(0.1)

            sys.stdout.write(" " * len(self.text) + '\r')

    def set_done(self):
        self.done.set()

    def change_text(self, new_text):
        sys.stdout.write(" " * len(self.text) + '\r')
        sys.stdout.flush()
        self.text = new_text
