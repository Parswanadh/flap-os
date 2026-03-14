"""Telegram bot interface for FLAP using aiogram long polling."""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandObject
from aiogram.types import Message
from dotenv import load_dotenv

from backend.agents.flap_core import build_chat_messages
from backend.memory.mem0_store import Mem0Store, MemoryStoreError
from backend.router.llm_router import LLMRouter
from backend.tools.computer_control import ComputerControl
from backend.tools.terminal_manager import TerminalManagerClient

load_dotenv()

DESTRUCTIVE_PATTERN = re.compile(
    r"\b(rm\s+-rf|shutdown|reboot|mkfs|dd\s+if=|:\(\)\{|poweroff|init\s+0)\b",
    re.IGNORECASE,
)


class TelegramBotError(RuntimeError):
    """Raised for configuration/runtime errors in Telegram integration."""


def is_destructive_command(command: str) -> bool:
    """Detect obviously destructive shell commands requiring explicit confirmation."""
    return bool(DESTRUCTIVE_PATTERN.search(command))


def parse_run_arguments(raw_args: str) -> tuple[bool, str]:
    """Parse /run arguments into (force, command)."""
    cleaned = raw_args.strip()
    if not cleaned:
        return (False, "")
    if cleaned.startswith("--force "):
        return (True, cleaned[len("--force ") :].strip())
    return (False, cleaned)


async def transcribe_groq_voice(
    *,
    audio_bytes: bytes,
    filename: str = "voice.ogg",
    model: str = "whisper-large-v3-turbo",
) -> str:
    """Transcribe audio bytes using Groq Whisper endpoint."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise TelegramBotError("GROQ_API_KEY is required for voice transcription.")

    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": (filename, audio_bytes, "audio/ogg")}
    data = {"model": model}
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
    if response.status_code != 200:
        raise TelegramBotError(f"Groq transcription failed ({response.status_code}): {response.text[:300]}")
    payload = response.json()
    text = str(payload.get("text", "")).strip()
    if not text:
        raise TelegramBotError("Groq transcription returned empty text.")
    return text


async def telegram_notify(text: str, *, chat_id: str | None = None, token: str | None = None) -> bool:
    """Send proactive Telegram notification from any module."""
    resolved_token = token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    resolved_chat = chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not resolved_token or not resolved_chat:
        return False

    bot = Bot(
        token=resolved_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    try:
        await bot.send_message(chat_id=resolved_chat, text=text)
    finally:
        await bot.session.close()
    return True


@dataclass
class FlapTelegramBot:
    """Telegram transport layer for FLAP."""

    router: LLMRouter
    memory_store: Mem0Store
    terminal_manager: TerminalManagerClient
    computer_control: ComputerControl
    token: str | None = None

    def __post_init__(self) -> None:
        bot_token = (self.token or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
        if not bot_token:
            raise TelegramBotError("TELEGRAM_BOT_TOKEN is missing.")
        self.bot = Bot(
            token=bot_token,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        )
        self.dp = Dispatcher()
        self._register_handlers()

    def _register_handlers(self) -> None:
        self.dp.message.register(self.handle_start, Command("start"))
        self.dp.message.register(self.handle_memory, Command("memory"))
        self.dp.message.register(self.handle_status, Command("status"))
        self.dp.message.register(self.handle_terminals, Command("terminals"))
        self.dp.message.register(self.handle_run, Command("run"))
        self.dp.message.register(self.handle_voice, F.voice)
        self.dp.message.register(self.handle_media, F.photo | F.document)
        self.dp.message.register(self.handle_text, F.text)

    async def start_polling(self) -> None:
        """Start aiogram long-polling loop."""
        await self.dp.start_polling(self.bot)

    async def _store_message_memory(self, *, chat_id: int, role: str, text: str) -> None:
        await self.memory_store.add_memory(
            text=f"{role}: {text}",
            source="telegram",
            metadata={"chat_id": chat_id, "role": role, "ts": datetime.now(timezone.utc).isoformat()},
        )

    async def _respond_via_router(self, *, chat_id: int, user_text: str) -> str:
        messages = build_chat_messages(user_text)
        mode = "fast_chat" if len(user_text) < 1400 else "reasoning"
        result = await self.router.generate(
            messages=messages,
            requested_mode=mode,
            metadata={"surface": "telegram"},
        )
        response_text = result.output_text.strip() or "I couldn't generate a response."
        try:
            await self._store_message_memory(chat_id=chat_id, role="user", text=user_text)
            await self._store_message_memory(chat_id=chat_id, role="assistant", text=response_text)
        except MemoryStoreError:
            response_text += "\n\n🔔 Memory sync unavailable right now."
        return response_text

    async def handle_start(self, message: Message) -> None:
        await message.answer("✅ FLAP Telegram bridge is online.")

    async def handle_memory(self, message: Message) -> None:
        memories = await self.memory_store.recent_memories(limit=5)
        if not memories:
            await message.answer("No memories stored yet.")
            return
        lines = [f"{idx+1}. [{item.source}] {item.text}" for idx, item in enumerate(memories)]
        await message.answer("\n".join(lines))

    async def handle_terminals(self, message: Message) -> None:
        sessions = await self.terminal_manager.list_sessions()
        lines = [
            f"- {session.name}: {'alive' if session.alive else 'down'} (pid={session.pid}, buffer={session.buffer_length})"
            for session in sessions
        ]
        await message.answer("Terminal sessions:\n" + ("\n".join(lines) if lines else "No sessions found."))

    async def handle_status(self, message: Message) -> None:
        docker = await self.computer_control.execute_shell(
            "docker ps --format '{{.Names}}:{{.Status}}'",
            timeout_s=20.0,
            check=False,
        )
        disk = await self.computer_control.execute_shell("df -h /", timeout_s=20.0, check=False)
        sessions = await self.terminal_manager.list_sessions()
        active_count = sum(1 for session in sessions if session.alive)
        status_text = (
            f"✅ Status\n"
            f"Active terminals: {active_count}/{len(sessions)}\n\n"
            f"Docker:\n{docker.stdout.strip() or docker.stderr.strip() or '(no output)'}\n\n"
            f"Disk:\n{disk.stdout.strip() or disk.stderr.strip() or '(no output)'}"
        )
        await message.answer(status_text)

    async def handle_run(self, message: Message, command: CommandObject) -> None:
        raw_args = command.args or ""
        force, shell_command = parse_run_arguments(raw_args)
        if not shell_command:
            await message.answer("Usage: /run [--force] <command>")
            return
        if is_destructive_command(shell_command) and not force:
            await message.answer(
                f"⚠️ Potentially destructive command blocked.\n"
                f"Resend with explicit confirmation:\n/run --force {shell_command}"
            )
            return
        result = await self.computer_control.execute_shell(shell_command, timeout_s=60.0, check=False)
        output = (result.stdout or result.stderr).strip()
        await message.answer(
            f"Exit: {result.exit_code}\n"
            f"Duration: {result.duration_ms}ms\n"
            f"```{output[:3500]}```"
        )

    async def handle_voice(self, message: Message) -> None:
        if message.voice is None:
            raise TelegramBotError("Voice handler invoked without voice payload.")
        file_info = await self.bot.get_file(message.voice.file_id)
        buffer = io.BytesIO()
        await self.bot.download(file=file_info.file_path, destination=buffer)  # type: ignore[arg-type]
        transcript = await transcribe_groq_voice(audio_bytes=buffer.getvalue(), filename=f"{message.voice.file_id}.ogg")
        response = await self._respond_via_router(chat_id=message.chat.id, user_text=transcript)
        await message.answer(f"🎤 {transcript}\n\n{response}")

    async def handle_media(self, message: Message) -> None:
        file_name = "media"
        byte_count = 0

        if message.photo:
            photo = message.photo[-1]
            file_info = await self.bot.get_file(photo.file_id)
            buffer = io.BytesIO()
            await self.bot.download(file=file_info.file_path, destination=buffer)  # type: ignore[arg-type]
            file_name = f"photo_{photo.file_id}.jpg"
            byte_count = len(buffer.getvalue())
        elif message.document:
            file_info = await self.bot.get_file(message.document.file_id)
            buffer = io.BytesIO()
            await self.bot.download(file=file_info.file_path, destination=buffer)  # type: ignore[arg-type]
            file_name = message.document.file_name or f"file_{message.document.file_id}"
            byte_count = len(buffer.getvalue())

        analysis_prompt = (
            f"User sent a media/file upload.\n"
            f"Name: {file_name}\n"
            f"Size bytes: {byte_count}\n"
            "Provide concise next-step analysis and what information to extract."
        )
        response = await self._respond_via_router(chat_id=message.chat.id, user_text=analysis_prompt)
        await message.answer(response)

    async def handle_text(self, message: Message) -> None:
        if message.text is None:
            raise TelegramBotError("Text handler invoked without text payload.")
        response = await self._respond_via_router(chat_id=message.chat.id, user_text=message.text)
        await message.answer(response)
