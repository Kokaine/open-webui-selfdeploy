"""
title: Deepseek R1 Manifold Pipe with Gemini Vision Support
authors: [MCode-Team, Ethan Copping, zgccrui]
author_url: [https://github.com/MCode-Team, https://github.com/CoppingEthan]
funding_url: https://github.com/open-webui
version: 0.1.7
license: MIT
environment_variables:
    - DEEPSEEK_API_KEY (required)
    - GOOGLE_API_KEY (required for image processing)

User: [Text + Image]
System:
1. Gemini reads the image and generates a description.  
2. Combines the image description with the text.  
3. Sends the combined content to DeepSeek for processing.  
4. DeepSeek responds back.

# Acknowledgments
Adapted code from [Ethan Copping] to add realtime preview of the thinking process for Deepseek R1
Adapted code from [zgccrui] to add Display the reasoning chain of the DeepSeek R1

"""

import os
import json
import time
import logging
import httpx
import re
# import google.generativeai as genai
from typing import (
    List,
    Union,
    Generator,
    Iterator,
    Dict,
    Optional,
    AsyncIterator,
    Tuple,
    Awaitable,
    Callable,
)
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message
from openai import OpenAI

class CacheEntry:
    def __init__(self, description: str):
        self.description = description
        self.timestamp = time.time()


class Pipe:
    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB per image
    TOTAL_MAX_IMAGE_SIZE = 100 * 1024 * 1024  # 100MB total
    REQUEST_TIMEOUT = (3.05, 60)
    CACHE_EXPIRATION = 30 * 60  # 30 minutes in seconds
    MODEL_MAX_TOKENS = {
        "deepseek-chat": 8192,
        "deepseek-reasoner": 8192,
        "deepseek-r1": 8192,
        "deepseek-v3": 8192
    }

    class Valves(BaseModel):
        DEEPSEEK_BASE_URL: str = Field(
            default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            description="Your DeepSeek Base URL",
        )
        DEEPSEEK_API_KEY: str = Field(
            default=os.getenv("DEEPSEEK_API_KEY", ""),
            description="Your DeepSeek API key",
        )
        DEEPSEEK_MODEL: str = Field(
            default=os.getenv("DEEPSEEK_MODEL", ""),
            description="Your DeepSeek Model",
        )
        VISION_API_KEY: str = Field(
            default=os.getenv("VISION_API_KEY", ""),
            description="Your Vision API key for image processing",
        )
        VISION_API_URL: str = Field(
            default=os.getenv("VISION_API_URL", ""),
            description="Your Vision API URL for image processing",
        )
        VISION_MODEL: str = Field(
            default=os.getenv("VISION_MODEL", ""),
            description="Your Vision Model Name for image processing",
        )

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.type = "manifold"
        self.id = "deepseek"
        self.name = "deepseek/"
        self.valves = self.Valves()
        self.request_id = None
        self.image_cache = {}

        self.clean_pattern = re.compile(r"<details>.*?</details>\n\n", flags=re.DOTALL)
        self.buffer_size = 3
        self.thinking_state = -1  # -1: Not started, 0: Thinking, 1: Answered

    @staticmethod
    def get_model_id(model_name: str) -> str:
        return model_name.replace(".", "/").split("/")[-1]

    def get_deepseek_models(self) -> List[Dict[str, str]]:
        try:
            headers = {
                "Authorization": f"Bearer {self.valves.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            }
            with httpx.Client() as client:
                response = client.get(
                    f"{self.valves.DEEPSEEK_BASE_URL}/models",
                    headers=headers,
                    timeout=10,
                )
            response.raise_for_status()
            models_data = response.json()
            return [
                {"id": model["id"], "name": model["id"]}
                for model in models_data.get("data", [])
            ]
        except Exception as e:
            logging.error(f"Error getting models: {e}")
            return []

    def pipes(self) -> List[dict]:
        return self.get_deepseek_models()

    def clean_expired_cache(self):
        current_time = time.time()
        expired_keys = [
            key
            for key, entry in self.image_cache.items()
            if current_time - entry.timestamp > self.CACHE_EXPIRATION
        ]
        for key in expired_keys:
            del self.image_cache[key]

    def extract_images_and_text(self, message: Dict) -> Tuple[List[Dict], str]:
        images = []
        text_parts = []
        content = message.get("content", "")

        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    images.append(item)
        else:
            text_parts.append(content)

        return images, " ".join(text_parts)

    async def process_image_with_gemini(
        self, image_data: Dict, __event_emitter__=None
    ) -> str:
        try:
            if not self.valves.GOOGLE_API_KEY:
                raise ValueError("VISION_API is required for image processing")

            self.clean_expired_cache()
            image_url = image_data.get("image_url", {}).get("url", "")
            image_key = image_url.split(",", 1)[1] if "," in image_url else image_url

            if image_key in self.image_cache:
                return self.image_cache[image_key].description

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "处理图片中...",
                            "done": False,
                        },
                    }
                )

            # genai.configure(api_key=self.valves.GOOGLE_API_KEY)
            # model = genai.GenerativeModel("gemini-2.0-flash")
            client = OpenAI(
                api_key=self.valves.VISION_API_KEY, # 混元 APIKey
                base_url=self.valves.VISION_API_URL, # 混元 endpoint
            )

            if image_url.startswith("data:image"):
                image_part = {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_url.split(",", 1)[1],
                    }
                }
            else:
                image_part = {"image_url": image_url}

            # response = model.generate_content(
            #     ["Describe this image in detail", image_part]
            # )
            response = client.chat.completions.create(
                model=self.valves.VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "contents": [
                            {
                                "type": "text",
                                "text": "请尽可能详细的描述这张图片的全部内容"
                            },
                            {
                                "type": "image_url",
                                "image_url": image_url
                            }
                        ]
                    },
                ],
            )
            description = response.text

            self.image_cache[image_key] = CacheEntry(description)
            if len(self.image_cache) > 100:
                oldest_key = min(
                    self.image_cache.keys(), key=lambda k: self.image_cache[k].timestamp
                )
                del self.image_cache[oldest_key]

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "图片处理完成",
                            "done": True,
                        },
                    }
                )

            return description

        except Exception as e:
            logging.error(f"Image processing error: {str(e)}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Image Error: {str(e)}",
                            "done": True,
                        },
                    }
                )
            return f"[Image Error: {str(e)}]"

    async def process_messages(
        self, messages: List[Dict], __event_emitter__=None
    ) -> List[Dict]:
        processed_messages = []
        for message in messages:
            images, text = self.extract_images_and_text(message)
            if images:
                image_descriptions = []
                for idx, image in enumerate(images, 1):
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"处理照片 {idx}/{len(images)}...",
                                    "done": False,
                                },
                            }
                        )

                    description = await self.process_image_with_gemini(
                        image, __event_emitter__
                    )

                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"Image {idx} analysis complete",
                                    "done": True,
                                },
                            }
                        )

                    image_descriptions.append(f"[Image Description: {description}]")

                combined_content = text + " " + " ".join(image_descriptions)
                processed_messages.append(
                    {"role": message["role"], "content": combined_content.strip()}
                )
            else:
                processed_messages.append(message)
        return processed_messages

    async def _stream_response(
        self,
        url: str,
        headers: dict,
        payload: dict,
        __event_emitter__=None,
        model_id: str = "",
    ) -> AsyncIterator[str]:
        buffer = []
        self.thinking_state = -1
        last_status_time = time.time()
        status_dots = 0

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.REQUEST_TIMEOUT,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            continue

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError as e:
                            logging.error(
                                f"Failed to parse data line: {data_str}, error: {e}"
                            )
                            continue

                        choice = data.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        reasoning = delta.get("reasoning_content") or ""
                        content = delta.get("content") or ""
                        finish_reason = choice.get("finish_reason")

                        if self.thinking_state == -1 and reasoning:
                            self.thinking_state = 0
                            buffer.append(
                                "<details>\n<summary>思考过程</summary>\n\n"
                            )
                            if __event_emitter__:
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": "思考中...",
                                            "done": False,
                                        },
                                    }
                                )

                        elif self.thinking_state == 0 and not reasoning and content:
                            self.thinking_state = 1
                            buffer.append("\n</details>\n\n")
                            if __event_emitter__:
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {"description": "", "done": True},
                                    }
                                )

                        if self.thinking_state == 0 and (model_id == "deepseek-reasoner" or model_id == "deepseek-r1"):
                            current_time = time.time()
                            if current_time - last_status_time > 1:
                                status_dots = (status_dots % 3) + 1
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": f"Thinking{'.'*status_dots}",
                                            "done": False,
                                        },
                                    }
                                )
                                last_status_time = current_time

                        if reasoning:
                            buffer.append(reasoning.replace("\n", "\n> "))
                        elif content:
                            buffer.append(content)

                        if finish_reason == "stop":
                            if self.thinking_state == 0:
                                buffer.append("\n</details>\n\n")
                            break

                        if len(buffer) >= self.buffer_size or "\n" in (
                            reasoning + content
                        ):
                            yield "".join(buffer)
                            buffer.clear()

                    if buffer:
                        yield "".join(buffer)

        except Exception as e:
            error_msg = f"Stream Error: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            yield error_msg

    async def _regular_request(
        self,
        url: str,
        headers: dict,
        payload: dict,
        __event_emitter__=None,
        model_id: str = "",
    ) -> dict:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()

                # Process DeepSeek response structure
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    message = choice.get("message", {})
                    original_content = message.get("content", "")
                    reasoning = message.get("reasoning_content", "")

                    # Combine reasoning and content
                    processed_content = original_content
                    if reasoning:
                        processed_content = (
                            f"<details>\n<summary>Thinking Process</summary>\n\n"
                            f"{reasoning}\n</details>\n\n{original_content}"
                        )
                        processed_content = self.clean_pattern.sub(
                            "", processed_content
                        ).strip()

                    # Modify response to match expected structure
                    data["choices"][0]["message"]["content"] = processed_content
                    data["choices"][0]["message"]["reasoning_content"] = reasoning

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": (
                                    data["choices"][0]["message"]["content"]
                                    if data.get("choices")
                                    else ""
                                ),
                                "done": True,
                            },
                        }
                    )

                return data

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logging.error(f"Regular request failed: {error_msg}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": error_msg,
                            "done": True,
                        },
                    }
                )
            return {"error": error_msg, "choices": []}

    async def pipe(
        self, body: Dict, __event_emitter__=None
    ) -> Union[AsyncIterator[str], dict]:
        if not self.valves.DEEPSEEK_API_KEY:
            error_msg = "Error: DEEPSEEK_API_KEY is required"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return {"error": error_msg, "choices": []}

        try:
            system_message, messages = pop_system_message(body.get("messages", []))
            processed_messages = await self.process_messages(
                messages, __event_emitter__
            )

            for msg in processed_messages:
                if msg.get("role") == "assistant" and "content" in msg:
                    msg["content"] = self.clean_pattern.sub("", msg["content"]).strip()

            model_id = self.get_model_id(body["model"])
            # model_id = self.valves.DEEPSEEK_MODEL
            max_tokens_limit = self.MODEL_MAX_TOKENS.get(model_id, 8192)

            if system_message:
                processed_messages.insert(
                    0, {"role": "system", "content": str(system_message)}
                )

            payload = {
                "model": model_id,
                "messages": processed_messages,
                "max_tokens": min(
                    body.get("max_tokens", max_tokens_limit), max_tokens_limit
                ),
                "temperature": float(body.get("temperature", 0.7)),
                "stream": body.get("stream", False),
            }

            headers = {
                "Authorization": f"Bearer {self.valves.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            }

            if payload["stream"]:
                return self._stream_response(
                    url=f"{self.valves.DEEPSEEK_BASE_URL}/chat/completions",
                    headers=headers,
                    payload=payload,
                    __event_emitter__=__event_emitter__,
                    model_id=model_id,
                )
            else:
                response_data = await self._regular_request(
                    url=f"{self.valves.DEEPSEEK_BASE_URL}/chat/completions",
                    headers=headers,
                    payload=payload,
                    __event_emitter__=__event_emitter__,
                    model_id=model_id,
                )

                # Ensure response structure consistency
                if "error" in response_data:
                    return response_data
                if not response_data.get("choices"):
                    return {"error": "Empty response from API", "choices": []}
                return response_data

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logging.error(f"Pipe processing failed: {error_msg}")
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return {"error": error_msg, "choices": []}
