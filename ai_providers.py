import json
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import httpx
from openai import OpenAI
import anthropic
import google.generativeai as genai
from starlette.websockets import WebSocketState

#logger = logging.get#logger(__name__)

# Global constant for model being used
CHAT_GPT_MODEL_BEING_USED = 'gpt-4o-mini'


class AIProvider(ABC):
    """
    Abstract base class for all AI providers
    Defines the interface that all concrete providers must implement
    """
    
    def __init__(self, api_key: str, websocket=None):
        self.api_key = api_key
        self.websocket = websocket
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=15.0, read=25.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )
        
    @abstractmethod
    async def translate(self, text: str, target_language: str, system_prompt: str) -> str:
        """Translate text to target language"""
        pass
    
    @abstractmethod
    async def translate_streaming(self, text: str, target_language: str, system_prompt: str, timestamp: float = None) -> None:
        """Translate text to target language with streaming response"""
        pass
        
    @abstractmethod
    async def summarize(self, text: str, system_prompt: str) -> str:
        """Generate summary of the text"""
        pass
        
    @abstractmethod
    async def extract_keywords(self, text: str, system_prompt: str) -> Dict[str, Any]:
        """Extract keywords from the text"""
        pass
        
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the provider"""
        pass
        
    @abstractmethod
    def validate_api_key(self) -> bool:
        """Validate the API key for this provider"""
        pass

    @abstractmethod
    def supports_streaming(self) -> bool:
        """Return whether this provider supports streaming"""
        pass

    @abstractmethod
    async def connect_to(self) -> None:
        """Process responses from AI provider"""
        pass
    
    def _parse_keywords_json(self, raw_content: str) -> Dict[str, Any]:
        """Helper method to parse JSON from AI provider responses"""
        try:
            # Remove markdown code block formatting if present
            if raw_content.strip().startswith('```'):
                # Extract JSON content between ```json and ```
                lines = raw_content.strip().split('\n')
                json_lines = []
                in_json_block = False
                
                for line in lines:
                    if line.strip().startswith('```json') or line.strip().startswith('```'):
                        in_json_block = not in_json_block
                        continue
                    if in_json_block:
                        json_lines.append(line)
                
                json_content = '\n'.join(json_lines)
            else:
                json_content = raw_content
            
            # Parse the JSON string into a dictionary
            keywords_dict = json.loads(json_content)
            #logger.info(f"Successfully parsed {len(keywords_dict.get('keywords', []))} keywords using {self.get_provider_name()}")
            return keywords_dict
            
        except json.JSONDecodeError as e:
            return {
                "keywords": [],
                "error": f"Failed to parse JSON: {str(e)}",
                "raw_content": raw_content
            }
        except Exception as e:
            #logger.error(f"Unexpected error parsing keywords response from {self.get_provider_name()}: {e}")
            return {
                "keywords": [],
                "error": f"Unexpected error: {str(e)}",
                "raw_content": raw_content
            }
    
    async def _send_error_to_websocket(self, message: str, error_code: str, severity: str = 'warning', persistent: bool = False):
        """Helper method to send errors to websocket with severity and persistence info"""
        if self.websocket:
            try:
                # Check websocket state like the original code did
                if self.websocket.client_state != WebSocketState.DISCONNECTED:
                    await self.websocket.send_json({
                        'type': 'translation_error',
                        'content': message,
                        'error_code': error_code,
                        'severity': severity,
                        'persistent': persistent
                    })
            except Exception as e:
                #logger.error(f"Failed to send error message to websocket: {e}")
                pass
    
    async def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'http_client'):
            await self.http_client.aclose()
        
        # Close WebSocket connections if they exist
        if hasattr(self, 'openai_ws') and self.openai_ws is not None:
            try:
                await self.openai_ws.close()
            except Exception:
                pass
        
        if hasattr(self, 'gemini_ws') and self.gemini_ws is not None:
            try:
                await self.gemini_ws.close()
            except Exception:
                pass


class OpenAIProvider(AIProvider):
    """OpenAI implementation of AIProvider"""
    
    def __init__(self, api_key: str, websocket=None, model_name: str = None):
        super().__init__(api_key, websocket)
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name or CHAT_GPT_MODEL_BEING_USED
        
    def get_provider_name(self) -> str:
        return "OpenAI"
        
    def supports_streaming(self) -> bool:
        return True
        
    def validate_api_key(self) -> bool:
        try:
            models = self.client.models.list()
            return True
        except Exception as e:
            #logger.error(f"OpenAI API key validation failed: {e}")
            return False
        
    
    async def connect_to(self) -> None:
        try:
                import websockets
                """Establish connection to OpenAI's realtime API."""
                # Use additional_headers for newer websockets versions
                try:
                    self.openai_ws = await websockets.connect(
                        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview',
                        additional_headers={
                            "Authorization": f"Bearer {self.openai_api_key}",
                            "OpenAI-Beta": "realtime=v1"
                        }
                    )
                except TypeError:
                    # Fallback for older websockets versions
                    self.openai_ws = await websockets.connect(
                        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview',
                        extra_headers={
                            "Authorization": f"Bearer {self.openai_api_key}",
                            "OpenAI-Beta": "realtime=v1"
                        }
                    )
                
                # Initialize session
                session_update = {
                    "type": "session.update",
                    "session": {
                        "instructions": self.system_prompt,
                        "modalities": ["text"],
                        "temperature": 0.6,
                    }
                }
                await self.openai_ws.send(json.dumps(session_update))
                # print("Connected to OpenAI")
        except Exception as e:
            print(f"Error connecting to OpenAI: {e}")
            raise e
        

    async def translate(self, text: str, target_language: str, system_prompt: str) -> str:
        """Translate text using OpenAI with connection retry logic"""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ],
                    "temperature": 0.3,
                    "stream": False
                }
                
                response = await self.http_client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    translation = response_data['choices'][0]['message']['content'].strip()
                    #logger.debug(f"OpenAI translation successful: {translation[:50]}...")
                    return translation
                else:
                    error_text = response.text
                    #logger.error(f"OpenAI API error {response.status_code}: {error_text}")
                    await self._handle_openai_error(response.status_code, error_text)
                    raise Exception(f"OpenAI API error: {response.status_code}")
                    
            except httpx.ConnectTimeout as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    #logger.warning(f"OpenAI connection timeout attempt {attempt + 1}/{max_retries}, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                    continue
                else:
                    #logger.error(f"OpenAI translation timeout after {max_retries} attempts: {e}")
                    raise Exception(f"Translation timeout: {str(e)}")
            except httpx.TimeoutException as e:
                #logger.error(f"OpenAI translation timeout: {e}")
                raise Exception(f"Translation timeout: {str(e)}")
            except httpx.HTTPError as e:
                #logger.error(f"OpenAI HTTP error: {e}")
                raise Exception(f"HTTP error: {str(e)}")
            except Exception as e:
                #logger.error(f"OpenAI translation error: {e}")
                raise Exception(f"Translation failed: {str(e)}")
    
    async def translate_streaming(self, text: str, target_language: str, system_prompt: str, timestamp: float = None) -> None:
        """Translate text using OpenAI with streaming response"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.3,
                "stream": True
            }
            
            # Send translation start signal
            if self.websocket:
                try:
                    await self.websocket.send_json({
                        'type': 'translation_start',
                        'content': f'Translating with OpenAI ({self.model_name}): {text[:30]}...' if len(text) > 30 else f'Translating with OpenAI: {text}',
                        'timestamp': timestamp
                    })
                except Exception:
                    pass
            
            async with self.http_client.stream(
                "POST", 
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                if response.status_code == 200:
                    full_translation = ""
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            chunk_data = line[6:]
                            if chunk_data == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(chunk_data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                
                                if content and self.websocket:
                                    full_translation += content
                                    try:
                                        await self.websocket.send_json({
                                            'type': 'assistant',
                                            'content': content,
                                            'timestamp': timestamp
                                        })
                                    except Exception:
                                        break
                            except json.JSONDecodeError:
                                continue
                    
                    # Send completion signal
                    if self.websocket:
                        try:
                            await self.websocket.send_json({
                                'type': 'assistant_done',
                                'content': 'Completed',
                                'timestamp': timestamp
                            })
                        except Exception:
                            pass
                            
                else:
                    error_text = await response.aread()
                    await self._handle_openai_error(response.status_code, error_text.decode())
                    
        except Exception as e:
            #logger.error(f"OpenAI streaming translation error: {e}")
            raise Exception(f"Streaming translation failed: {str(e)}")
    
    async def summarize(self, text: str, system_prompt: str) -> str:
        """Generate summary using OpenAI"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.3,
                "stream": False
            }
            
            response = await self.http_client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                await self._handle_openai_error(response.status_code, response.text)
                return None
                
        except Exception as e:
            #logger.error(f"OpenAI summarization error: {e}")
            return None
    
    async def extract_keywords(self, text: str, system_prompt: str) -> Dict[str, Any]:
        """Extract keywords using OpenAI"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.3,
                "stream": False
            }
            
            response = await self.http_client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                raw_content = response.json()['choices'][0]['message']['content']
                return self._parse_keywords_json(raw_content)
            else:
                await self._handle_openai_error(response.status_code, response.text)
                return {"keywords": [], "error": f"API error: {response.status_code}"}
                
        except Exception as e:
            #logger.error(f"OpenAI keyword extraction error: {e}")
            return {"keywords": [], "error": f"OpenAI error: {str(e)}"}
    
    async def _handle_openai_error(self, status_code: int, response_text: str):
        """Handle OpenAI specific errors with proper severity classification"""
        # Parse response for detailed error information
        try:
            error_data = json.loads(response_text) if response_text else {}
            error_message = error_data.get('error', {}).get('message', 'Unknown error')
            error_type = error_data.get('error', {}).get('type', 'unknown_error')
        except json.JSONDecodeError:
            error_message = response_text or 'Unknown error'
            error_type = 'unknown_error'
        
        # Determine severity and appropriate message
        if status_code == 429:
            if 'quota' in error_message.lower() or 'insufficient_quota' in error_type:
                await self._send_error_to_websocket(
                    'OpenAI API quota has been exceeded. Please check your OpenAI billing and add credits to continue using translation features.',
                    'TRANSLATION_QUOTA_EXCEEDED',
                    severity='critical',
                    persistent=True
                )
            else:
                await self._send_error_to_websocket(
                    'OpenAI API rate limit exceeded. Please wait a moment and try again.',
                    'TRANSLATION_RATE_LIMIT',
                    severity='warning',
                    persistent=False
                )
        elif status_code == 401:
            await self._send_error_to_websocket(
                'OpenAI API key is invalid or unauthorized. Please check your API key configuration.',
                'TRANSLATION_AUTH_ERROR',
                severity='critical',
                persistent=True
            )
        elif status_code == 402:
            await self._send_error_to_websocket(
                'OpenAI API payment required. Your account may be out of credits. Please check your billing settings.',
                'TRANSLATION_QUOTA_EXCEEDED',
                severity='critical',
                persistent=True
            )
        elif status_code == 403:
            await self._send_error_to_websocket(
                'OpenAI API access forbidden. Please verify your API key permissions.',
                'TRANSLATION_AUTH_ERROR',
                severity='critical',
                persistent=True
            )
        elif status_code >= 500:
            await self._send_error_to_websocket(
                'OpenAI API is experiencing server issues. Please try again later.',
                'TRANSLATION_TEMPORARY_ERROR',
                severity='warning',
                persistent=False
            )
        else:
            await self._send_error_to_websocket(
                f'OpenAI API error (Status {status_code}): {error_message}',
                'TRANSLATION_TEMPORARY_ERROR',
                severity='warning',
                persistent=False
            )


class ClaudeProvider(AIProvider):
    """Claude (Anthropic) implementation of AIProvider"""
    
    def __init__(self, api_key: str, websocket=None):
        super().__init__(api_key, websocket)
        self.client = anthropic.Anthropic(api_key=api_key)
        
    def get_provider_name(self) -> str:
        return "Claude"
        
    def supports_streaming(self) -> bool:
        return True
        
    def validate_api_key(self) -> bool:
        try:
            # Claude doesn't have a simple validation endpoint, so we'll do a basic check
            return bool(self.api_key and len(self.api_key) > 20)
        except Exception as e:
            #logger.error(f"Claude API key validation failed: {e}")
            return False
    
    async def connect_to(self) -> None:
        """Establish connection to Claude's realtime API."""
        try:
            import websockets
            
            # Note: Claude doesn't have a public WebSocket API like OpenAI or Gemini
            # This method is included for consistency but will raise a NotImplementedError
            # In the future, if Claude releases a WebSocket API, this can be implemented
            
            raise NotImplementedError(
                "Claude (Anthropic) does not currently provide a public WebSocket API. "
                "This method is included for interface consistency. "
                "Use the regular HTTP API methods for Claude integration."
            )
            
        except Exception as e:
            print(f"Error connecting to Claude API: {e}")
            raise e
    
    async def translate(self, text: str, target_language: str, system_prompt: str) -> str:
        """Translate text using Claude"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    temperature=0.3,
                    system=system_prompt,
                    messages=[{"role": "user", "content": text}]
                )
            )
            
            if response and response.content:
                translation = response.content[0].text.strip()
                return translation
            else:
                raise Exception("Empty response from Claude")
                
        except Exception as e:
            #logger.error(f"Claude translation error: {e}")
            await self._handle_claude_error(str(e))
            raise
    
    async def translate_streaming(self, text: str, target_language: str, system_prompt: str, timestamp: float = None) -> None:
        """Translate text using Claude with streaming response"""
        try:
            # Send translation start signal
            if self.websocket:
                try:
                    await self.websocket.send_json({
                        'type': 'translation_start',
                        'content': f'Translating with Claude: {text[:30]}...' if len(text) > 30 else f'Translating with Claude: {text}',
                        'timestamp': timestamp
                    })
                except Exception:
                    pass
            
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 4000,
                "system": system_prompt,
                "messages": [{"role": "user", "content": text}],
                "stream": True
            }
            
            async with self.http_client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers
            ) as response:
                if response.status_code == 200:
                    full_translation = ""
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            chunk_data = line[6:]
                            if chunk_data == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(chunk_data)
                                if chunk.get("type") == "content_block_delta":
                                    delta = chunk.get("delta", {})
                                    content = delta.get("text", "")
                                    
                                    if content and self.websocket:
                                        full_translation += content
                                        try:
                                            await self.websocket.send_json({
                                                'type': 'assistant',
                                                'content': content,
                                                'timestamp': timestamp
                                            })
                                        except Exception:
                                            break
                            except json.JSONDecodeError:
                                continue
                    
                    # Send completion signal
                    if self.websocket:
                        try:
                            await self.websocket.send_json({
                                'type': 'assistant_done',
                                'content': 'Completed',
                                'timestamp': timestamp
                            })
                        except Exception:
                            pass
                            
                else:
                    error_text = await response.aread()
                    await self._handle_claude_error(error_text.decode())
                    
        except Exception as e:
            #logger.error(f"Claude streaming translation error: {e}")
            await self._handle_claude_error(str(e))
            raise Exception(f"Streaming translation failed: {str(e)}")
    
    async def summarize(self, text: str, system_prompt: str) -> str:
        """Generate summary using Claude"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    temperature=0.3,
                    system=system_prompt,
                    messages=[{"role": "user", "content": text}]
                )
            )
            return response.content[0].text
        except Exception as e:
            #logger.error(f"Claude summarization error: {e}")
            await self._handle_claude_error(str(e))
            return None
    
    async def extract_keywords(self, text: str, system_prompt: str) -> Dict[str, Any]:
        """Extract keywords using Claude"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=2000,
                    temperature=0.3,
                    system=system_prompt,
                    messages=[{"role": "user", "content": text}]
                )
            )
            
            raw_content = response.content[0].text
            return self._parse_keywords_json(raw_content)
            
        except Exception as e:
            #logger.error(f"Claude keyword extraction error: {e}")
            await self._handle_claude_error(str(e))
            return {"keywords": [], "error": f"Claude error: {str(e)}"}
    
    async def _handle_claude_error(self, error_message: str):
        """Handle Claude specific errors"""
        error_lower = error_message.lower()
        if "rate limit" in error_lower or "quota" in error_lower:
            await self._send_error_to_websocket(
                'Claude API rate limit exceeded or credits exhausted. Please check your Anthropic account and usage limits.',
                'CLAUDE_RATE_LIMIT'
            )
        elif "unauthorized" in error_lower or "api key" in error_lower:
            await self._send_error_to_websocket(
                'Claude API key is invalid or unauthorized. Please check your API key configuration.',
                'CLAUDE_AUTH_ERROR'
            )
        elif "payment" in error_lower or "billing" in error_lower:
            await self._send_error_to_websocket(
                'Claude API payment issue. Please check your Anthropic account billing settings.',
                'CLAUDE_PAYMENT_ERROR'
            )
        else:
            await self._send_error_to_websocket(
                f'Claude API error: {error_message}',
                'CLAUDE_ERROR'
            )


class GeminiProvider(AIProvider):
    """Google Gemini implementation of AIProvider"""


    '''
    models/gemini-1.5-flash
    models/gemini-1.5-pro
    models/gemini-1.5-pro-latest
    models/gemini-1.5-flash-latest
    models/gemini-1.5-pro-preview-0514
    models/gemini-1.5-flash-preview-0514
    '''
    
    def __init__(self, api_key: str, websocket=None, model_name: str = 'gemini-1.5-pro-latest'):
        super().__init__(api_key, websocket)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        
    def get_provider_name(self) -> str:
        return "Gemini"
        
    def supports_streaming(self) -> bool:
        return True
        
    def validate_api_key(self) -> bool:
        try:
            genai.configure(api_key=self.api_key)
            models = list(genai.list_models())
            return True
        except Exception as e:
            return False
    
    async def translate(self, text: str, target_language: str, system_prompt: str) -> str:
        """Translate text using Gemini"""
        try:
            prompt = f"{system_prompt}\n\nText to translate: {text}"
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.model.generate_content, prompt
            )
            
            if response and response.text:
                translation = response.text.strip()
                # Clean up any markdown formatting
                translation = translation.replace('*', '').replace('#', '').strip()
                return translation
            else:
                raise Exception("Empty response from Gemini")
                
        except Exception as e:
            await self._handle_gemini_error(str(e))
            raise

    async def connect_to(self) -> None:
        """Establish connection to Gemini's realtime API."""
        try:
            import websockets
            
            # WebSocket endpoint for Gemini Live API with API key as query parameter
            # According to https://ai.google.dev/api/live
            ws_url = f'wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={self.api_key}'
                        
            # Connect to WebSocket (no headers needed when using query parameter)
            try:
                self.gemini_ws = await websockets.connect(ws_url)
            except TypeError:
                # Fallback for older websockets versions
                self.gemini_ws = await websockets.connect(ws_url)
            
            print(f"WebSocket connection established: {self.gemini_ws}")
            
            # Send initial session configuration according to Live API docs
            session_config = {
                "setup": {
                    "model": self.model_name,
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 4000,
                        "responseModalities": ["text"]
                    },
                    "systemInstruction": "You are a helpful translator. Translate the provided text accurately."
                }
            }
            
            await self.gemini_ws.send(json.dumps(session_config))
                        
        except Exception as e:
            print(f"Error connecting to Gemini Live API: {e}")
            raise e
    
    async def send_translation_request(self, text: str, system_prompt: str) -> None:
        """Send a translation request through the Gemini WebSocket connection."""
        if not hasattr(self, 'gemini_ws') or self.gemini_ws is None:
            raise Exception("WebSocket connection not established. Call connect_to() first.")
        
        try:
            # Prepare the message according to Gemini Live API format
            # https://ai.google.dev/api/live#websocket-connection
            client_content = {
                "clientContent": {
                    "turns": [
                        {
                            "parts": [
                                {"text": f"{system_prompt}\n\nText to translate: {text}"}
                            ]
                        }
                    ],
                    "turnComplete": True
                }
            }
            
            print(f"Sending translation request to Gemini Live API: {client_content}")
            await self.gemini_ws.send(json.dumps(client_content))
            print(f"Sent translation request to Gemini: {text[:50]}...")
            
        except Exception as e:
            print(f"Error sending translation request to Gemini: {e}")
            raise e
    
    async def send_message(self, message: str, system_prompt: str = None) -> None:
        """Send a message through the Gemini WebSocket connection."""
        if not hasattr(self, 'gemini_ws') or self.gemini_ws is None:
            raise Exception("WebSocket connection not established. Call connect_to() first.")
        
        try:
            # Prepare the message according to Gemini Live API format
            client_content = {
                "clientContent": {
                    "turns": [
                        {
                            "parts": [
                                {"text": f"{system_prompt}\n\n{message}" if system_prompt else message}
                            ]
                        }
                    ],
                    "turnComplete": True
                }
            }
            
            await self.gemini_ws.send(json.dumps(client_content))
            
        except Exception as e:
            print(f"Error sending message to Gemini: {e}")
            raise e
    
    async def receive_response(self) -> str:
        """Receive response from Gemini WebSocket connection."""
        if not hasattr(self, 'gemini_ws') or self.gemini_ws is None:
            raise Exception("WebSocket connection not established. Call connect_to() first.")
        
        try:
            full_response = ""
            async for message in self.gemini_ws:
                data = json.loads(message)
                
                # Check for different message types
                if "content" in data:
                    # Regular content response
                    content = data["content"]
                    if "parts" in content:
                        for part in content["parts"]:
                            if "text" in part:
                                full_response += part["text"]
                
                elif "candidates" in data:
                    # Candidate response
                    candidates = data["candidates"]
                    if candidates and "content" in candidates[0]:
                        content = candidates[0]["content"]
                        if "parts" in content:
                            for part in content["parts"]:
                                if "text" in part:
                                    full_response += part["text"]
                
                # Check if response is complete
                if data.get("finishReason") or "usageMetadata" in data:
                    break
            
            return full_response.strip()
            
        except Exception as e:
            print(f"Error receiving response from Gemini: {e}")
            raise e
    
    async def close_connection(self) -> None:
        """Close the Gemini WebSocket connection."""
        if hasattr(self, 'gemini_ws') and self.gemini_ws is not None:
            try:
                await self.gemini_ws.close()
                self.gemini_ws = None
                print("Gemini WebSocket connection closed")
            except Exception as e:
                print(f"Error closing Gemini WebSocket connection: {e}")
                raise e
    
    async def translate_streaming(self, text: str, target_language: str, system_prompt: str, timestamp: float = None) -> None:
        """Translate text using Gemini with streaming response"""
        try:
            # Send translation start signal
            if self.websocket:
                try:
                    await self.websocket.send_json({
                        'type': 'translation_start',
                        'content': f'Translating with Gemini ({self.model_name}): {text[:30]}...' if len(text) > 30 else f'Translating with Gemini: {text}',
                        'timestamp': timestamp
                    })
                except Exception:
                    pass
            
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": f"{system_prompt}\n\nText to translate: {text}"}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 4000
                }
            }
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:streamGenerateContent?key={self.api_key}&alt=sse"
            
            async with self.http_client.stream(
                "POST",
                url,
                json=payload,
                headers=headers
            ) as response:
                if response.status_code == 200:
                    full_translation = ""
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            chunk_data = line[6:]
                            if chunk_data.strip() == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(chunk_data)
                                candidates = chunk.get("candidates", [])
                                if candidates:
                                    content = candidates[0].get("content", {})
                                    parts = content.get("parts", [])
                                    if parts:
                                        text_content = parts[0].get("text", "")
                                        
                                        if text_content and self.websocket:
                                            # Clean up any markdown formatting
                                            clean_content = text_content.replace('*', '').replace('#', '').strip()
                                            full_translation += clean_content
                                            try:
                                                await self.websocket.send_json({
                                                    'type': 'assistant',
                                                    'content': clean_content,
                                                    'timestamp': timestamp
                                                })
                                            except Exception:
                                                break
                            except json.JSONDecodeError:
                                continue
                    
                    # Send completion signal
                    if self.websocket:
                        try:
                            await self.websocket.send_json({
                                'type': 'assistant_done',
                                'content': 'Completed',
                                'timestamp': timestamp
                            })
                        except Exception:
                            pass
                            
                else:
                    error_text = await response.aread()
                    error_message = await self._parse_gemini_error(response.status_code, error_text.decode())
                    await self._handle_gemini_error(error_message)
                    
        except Exception as e:
            #logger.error(f"Gemini streaming translation error: {e}")
            await self._handle_gemini_error(str(e))
            raise Exception(f"Streaming translation failed: {str(e)}")
    
    async def summarize(self, text: str, system_prompt: str) -> str:
        """Generate summary using Gemini"""
        try:
            full_prompt = f"{system_prompt}\n\nText to summarize: {text}"
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.model.generate_content, full_prompt
            )
            
            if response and response.text:
                summary = response.text.strip()
                # Clean up any markdown formatting
                summary = summary.replace('*', '').replace('#', '').strip()
                return summary
            else:
                #logger.warning("Empty response from Gemini summarizer")
                return None
        except Exception as e:
            #logger.error(f"Gemini summarization error: {e}")
            await self._handle_gemini_error(str(e))
            return None
    
    async def extract_keywords(self, text: str, system_prompt: str) -> Dict[str, Any]:
        """Extract keywords using Gemini"""
        try:
            full_prompt = f"{system_prompt}\n\nText to extract keywords from: {text}"
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.model.generate_content, full_prompt
            )
            
            if response and response.text:
                raw_content = response.text.strip()
                return self._parse_keywords_json(raw_content)
            else:
                #logger.warning("Empty response from Gemini keywords extractor")
                return {"keywords": [], "error": "Empty response from Gemini"}
        except Exception as e:
            #logger.error(f"Gemini keyword extraction error: {e}")
            await self._handle_gemini_error(str(e))
            return {"keywords": [], "error": f"Gemini error: {str(e)}"}
    
    async def _parse_gemini_error(self, status_code: int, error_text: str) -> str:
        """Parse Gemini API error response to extract meaningful error message"""
        try:
            # Debug: Print raw error response
            print(f"DEBUG - Gemini raw error response (status {status_code}): '{error_text}'")
            print(f"DEBUG - Error text length: {len(error_text)}, is empty: {not error_text.strip()}")
            
            # Try to parse as JSON first
            if error_text.strip():
                try:
                    error_json = json.loads(error_text)
                    
                    # Gemini error structure: {"error": {"code": 400, "message": "...", "status": "..."}}
                    if "error" in error_json:
                        error_info = error_json["error"]
                        message = error_info.get("message", "")
                        code = error_info.get("code", status_code)
                        status = error_info.get("status", "")
                        
                        # Combine available error information
                        error_parts = []
                        if message:
                            error_parts.append(message)
                        if status and status != message:
                            error_parts.append(f"Status: {status}")
                        if code and code != status_code:
                            error_parts.append(f"Code: {code}")
                        
                        if error_parts:
                            return " | ".join(error_parts)
                    
                    # Fallback: look for any "message" field at root level
                    if "message" in error_json:
                        return error_json["message"]
                        
                except json.JSONDecodeError:
                    pass
            
            # If JSON parsing fails or no meaningful message found, use status code and raw text
            if error_text.strip():
                return f"HTTP {status_code}: {error_text.strip()[:200]}"  # Limit to 200 chars
            else:
                # Handle common HTTP status codes with empty responses
                if status_code == 400:
                    return "HTTP 400: Bad Request (possibly invalid model or parameters)"
                elif status_code == 401:
                    return "HTTP 401: Unauthorized (invalid API key)"
                elif status_code == 403:
                    return "HTTP 403: Forbidden (insufficient permissions or quota exceeded)"
                elif status_code == 404:
                    return "HTTP 404: Not Found (model not available or invalid endpoint)"
                elif status_code == 429:
                    return "HTTP 429: Too Many Requests (rate limit exceeded)"
                elif status_code == 500:
                    return "HTTP 500: Internal Server Error (Gemini service issue)"
                else:
                    return f"HTTP {status_code}: Unknown error (empty response)"
                
        except Exception as e:
            return f"HTTP {status_code}: Error parsing response - {str(e)}"

    async def _handle_gemini_error(self, error_message: str):
        """Handle Gemini specific errors"""
        error_lower = error_message.lower()
        if "rate limit" in error_lower or "quota" in error_lower or "resource_exhausted" in error_lower:
            await self._send_error_to_websocket(
                'Gemini API rate limit exceeded or quota exhausted. Please check your Google Cloud account and usage limits.',
                'GEMINI_RATE_LIMIT'
            )
        elif "unauthorized" in error_lower or "api key" in error_lower or "forbidden" in error_lower or "permission_denied" in error_lower:
            await self._send_error_to_websocket(
                'Gemini API key is invalid or unauthorized. Please check your API key configuration.',
                'GEMINI_AUTH_ERROR'
            )
        elif "payment" in error_lower or "billing" in error_lower or "insufficient" in error_lower:
            await self._send_error_to_websocket(
                'Gemini API payment issue. Please check your Google Cloud billing settings.',
                'GEMINI_PAYMENT_ERROR'
            )
        elif "not found" in error_lower or "404" in error_message:
            await self._send_error_to_websocket(
                'Gemini API model not found. Please check if the model name is correct and available.',
                'GEMINI_MODEL_NOT_FOUND'
            )
        elif "invalid" in error_lower and "request" in error_lower:
            await self._send_error_to_websocket(
                f'Gemini API invalid request: {error_message}',
                'GEMINI_INVALID_REQUEST'
            )
        else:
            print(f"Gemini API error: {error_message}")
            await self._send_error_to_websocket(
                f'Gemini API error: {error_message}' if error_message.strip() else 'Gemini API unknown error',
                'GEMINI_ERROR'
            )


class AIProviderFactory:
    """
    Factory class to create AI provider instances
    Implements the Factory design pattern for easy provider creation
    """
    
    _providers = {
        'openai': OpenAIProvider,
        'claude': ClaudeProvider,
        'gemini': GeminiProvider
    }
    
    @classmethod
    def create_provider(cls, provider_name: str, api_key: str, websocket=None, **kwargs) -> AIProvider:
        """
        Create and return an AI provider instance
        
        Args:
            provider_name: Name of the provider ('openai', 'claude', 'gemini')
            api_key: API key for the provider
            websocket: WebSocket connection for error reporting
            **kwargs: Additional provider-specific arguments
        
        Returns:
            AIProvider instance
        
        Raises:
            ValueError: If provider_name is not supported
        """
        provider_name = provider_name.lower()
        
        if provider_name not in cls._providers:
            raise ValueError(f"Unsupported AI provider: {provider_name}. Supported providers: {list(cls._providers.keys())}")
        
        provider_class = cls._providers[provider_name]
        
        try:
            # Create provider instance with appropriate arguments
            if provider_name in ['openai', 'gemini'] and 'model_name' in kwargs:
                return provider_class(api_key, websocket, kwargs['model_name'])
            else:
                return provider_class(api_key, websocket)
        except Exception as e:
            #logger.error(f"Failed to create {provider_name} provider: {e}")
            raise
    
    @classmethod
    def get_supported_providers(cls) -> List[str]:
        """Return list of supported provider names"""
        return list(cls._providers.keys())
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """
        Register a new AI provider class
        Allows for dynamic addition of new providers
        
        Args:
            name: Name of the provider
            provider_class: Class that implements AIProvider interface
        """
        if not issubclass(provider_class, AIProvider):
            raise ValueError(f"Provider class must inherit from AIProvider")
        
        cls._providers[name.lower()] = provider_class
        #logger.info(f"Registered new AI provider: {name}")
