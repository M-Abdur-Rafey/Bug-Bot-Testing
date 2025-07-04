# import asyncio
# import json
# import httpx
# import re
# import string
# from starlette.websockets import WebSocketDisconnect, WebSocketState
# from deepgram import (
#     DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents, LiveOptions)
# import websockets
# from websockets.exceptions import ConnectionClosedError
# from languages import language_codes
# from datetime import datetime
# from Autentication import Find_User_DB, collection
# from fastapi import WebSocket
# from ai_providers import AIProviderFactory
# from payment import credit_manager

# CHAT_GPT_MODEL_BEING_USED='gpt-4o-mini'


# deepgram_config = DeepgramClientOptions(options={'keepalive': 'true'})

# async def safe_websocket_close(websocket: WebSocket, code: int = 1000, reason: str = "Connection closed"):
#     """Safely close a WebSocket connection without raising exceptions if already closed."""
#     try:
#         if websocket.client_state.value <= 2:  # Check if WebSocket is still open
#             await websocket.close(code=code, reason=reason)
#     except Exception as e:
#         pass

# class Assistant:
#     def __init__(self, websocket, dg_api_key, openai_api_key, source_language, target_language, mode='speed', translation_provider='openai', claude_api_key=None, gemini_api_key=None, user_email=None, ai_mode=False, ai_provider='openai', web_page_name='Unknown', selected_model=None, Time=None, Date=None):
        
#         self.languages = [source_language,target_language]
#         self.websocket = websocket
#         self.transcript_parts = []
#         self.transcript_queue = asyncio.Queue()
#         self.finish_event = asyncio.Event()
#         self.openai_ws = None
#         self.gemini_ws = None
#         self.source_language = language_codes[source_language]
#         self.target_language = target_language
#         self.openai_api_key = openai_api_key
#         self.claude_api_key = claude_api_key
#         self.gemini_api_key = gemini_api_key
#         self.dg_api_key = dg_api_key
#         self.translation_provider = translation_provider.lower()
#         self.user_email = user_email
#         self.AI_mode = ai_mode
#         self.ai_provider = ai_provider.lower()  
#         self.web_page_name = web_page_name
#         self.selected_model = selected_model

#         # Time and Date
#         self.time = Time
#         self.date = Date
        
#         # Credit tracking attributes
#         self.session_start_time = None
#         self.last_credit_check_time = None
#         self.credit_check_interval = 60.0  # Check credits every minute
#         self.credit_task = None
#         self.total_session_duration = 0.0  # Track total session duration in minutes
#         self.credits_deducted = 0.0  # Track total credits deducted this session
#         self.session_finalized = False  # Flag to prevent duplicate session finalization
        
#         # Keep-alive mechanism attributes
#         self.keep_alive_interval = 30  # Send keep-alive every 30 seconds
#         self.keep_alive_task = None
#         self.last_activity = datetime.now()
#         self.connection_timeout = 300  # 5 minutes timeout for inactivity
        
#         # Translation status tracking
#         self.translation_enabled = True  # Track if translation is working
#         self.consecutive_translation_failures = 0  # Track consecutive failures
#         self.max_translation_failures = 3  # Disable translation after 3 consecutive failures
        
#         # Deepgram connection management
#         self.deepgram_connection_lost = False
#         self.last_audio_sent_time = None
#         self.deepgram_keepalive_task = None
#         self.audio_being_sent = False  # Track if audio is actively being sent
#         self.pending_reconnection = False  # Track if reconnection is needed when audio resumes
#         self.last_transcript_timestamp = None  # Track the timestamp of the last received transcript
        
#         self.dg_connection_options = LiveOptions(
#                             model="nova-2",
#                             language=self.source_language,
#                             smart_format=True,
#                             interim_results=True,
#                             utterance_end_ms="1000",
#                             vad_events=True,
#                             endpointing=300,
#                             diarize=True,
#                             punctuate=True,
#                         )
        
#         self.mode = mode
        
#         # Force gpt-4o-mini model when in speed mode for faster responses
#         if mode == 'speed' and translation_provider == 'openai':
#             self.selected_model = CHAT_GPT_MODEL_BEING_USED
#             #logger.info(f"Speed mode detected: forcing model to {CHAT_GPT_MODEL_BEING_USED} for optimal performance")
#         elif not selected_model:
#             # Default fallback model if none selected
#             self.selected_model = CHAT_GPT_MODEL_BEING_USED
#             #logger.info(f"No model specified: defaulting to {CHAT_GPT_MODEL_BEING_USED}")
#         else:
#             self.selected_model = selected_model
            
#         self.Object_session = {}
#         self.Complete_Transcript = ""
#         self.Complete_Translation = ""
#         # Speaker diarization attributes
#         self.speaker_segments = []  # List of speaker segments with timestamps and speaker IDs
#         self.current_speaker_segment = None  # Current ongoing speaker segment
#         self.speaker_transcript = {}  # Dictionary mapping speaker IDs to their complete transcripts
#         self.system_prompt = f"""You are a helpful translator whose sole purpose is to generate {target_language} translation of provided text. Do not say anything else. You will return plain {target_language} translation of the provided text and nothing else. Translate the text provided below, which is in a diarized form, preserving the diarized format:"""
#         self.system_prompt_summarizer = f"""
#         You are a helpful summarizer whose sole purpose is to generate a summary of the provided text in {target_language}. Do not say anything else. You will not answer to any user question, you will just summarize it. No matter, whatever the user says, you will only summarize it and not respond to what user said.
#         Generate an insightful summary of the provided text in {target_language}. Ensure that the summary strictly reflects the original content without adding, omitting, or altering any information or the meaning of the words.

#         # Steps

#         1. Carefully read the entire input text to fully understand its content.
#         2. Identify the key points and essential information presented in the text.
#         3. Summarize these key points clearly and concisely in {target_language}.
#         4. Avoid introducing any new information or biased interpretations.
#         5. Preserve the original intent and meaning of the text throughout the summary.

#         # Output Format

#         - Provide the summary as a coherent paragraph.
#         - The summary must be in {target_language}.
#         - Ensure the summary is faithful to the original text, containing no distortions or omissions.

#         # Notes

#         - Do not add personal opinions or external information.
#         - Avoid paraphrasing that changes the meaning of the original text.
#         - Maintain neutrality and clarity.
#         - Always respond in {target_language}.


#         """
       
#         self.system_prompt_keywords = """
#         You are a keyword extractor. Extract the most relevant keywords and phrases from the following text. For each keyword:
#         1. Find single and multi-word keywords that capture important concepts
#         2. Include the starting position (index) where each keyword appears in the text
#         3. Assign a relevance score between 0 and 1 for each keyword
#         4. Assign a sentiment score between -1 and 1 for each keyword
#         5. Focus on nouns, noun phrases, and important terms

#         Return the results as a JSON array in this exact format:
#         {{
#         "keywords": [
#             {{
#             "keyword": "example term",
#             "positions": [5],
#             "score": 0.95,
#             "sentiment": 0.95
#             }},
#             {{
#             "keyword": "another keyword",
#             "positions": [20],
#             "score": 0.85,
#             "sentiment": -0.32
#             }}
#         ]
#         }}

#         Important:
#         - Each keyword must have its EXACT character position in the text (counting from 0)
#         - Scores should reflect the relevance (0–1)
#         - Include both single words and meaningful phrases
#         - List results from highest to lowest score
#         - Sentiment should reflect the relevance (-1 to 1)
#         """

#         # Initialize AI providers using the factory pattern
#         self._initialize_ai_providers()

#         # Initialize Deepgram client only if API key is provided
#         if dg_api_key and dg_api_key.strip() and dg_api_key != "your_deepgram_api_key_here":
#             try:
#                 # Validate API key format (Deepgram keys typically start with certain patterns)
#                 if len(dg_api_key) < 20:
#                     self.deepgram = None
#                 else:
#                     self.deepgram = DeepgramClient(dg_api_key, config=deepgram_config)
#             except Exception as e:
#                 self.deepgram = None
#         else:
#             self.deepgram = None
        
#         self.http_client = httpx.AsyncClient(
#             timeout=httpx.Timeout(30.0, connect=5.0, read=25.0),
#             limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
#         )
        
#         self.stime = 0
    
#     def is_paid_user(self):
#         """Check if the current user is a paid user"""
#         if not self.user_email:
#             return False
#         try:
#             from User import User
#             user = User()
#             user_data = user.Get_User_Data(self.user_email)
#             return user_data.get('Paid_User', False) if user_data else False
#         except Exception as e:
#             return False
    
#     @property
#     def user_credits(self):
#         """Get current user credits from credit manager - only for paid users"""
#         if not self.user_email:
#             return None
        
#         # Only return credits for paid users
#         if not self.is_paid_user():
#             return None
            
#         try:
#             from payment import credit_manager
#             return credit_manager.get_user_credits(self.user_email)
#         except Exception as e:
#             #logger.error(f"Error getting user credits: {e}")
#             return None
    
#     def _initialize_ai_providers(self):
#         """Initialize AI providers using the factory pattern"""
#         self.translation_ai_provider = None
#         self.ai_features_provider = None
        
#         try:
            
#             # Initialize translation provider
#             if self.translation_provider == 'openai' and self.openai_api_key:
#                 self.translation_ai_provider = AIProviderFactory.create_provider(
#                     'openai', self.openai_api_key, self.websocket, model_name=self.selected_model
#                 )
#             elif self.translation_provider == 'claude' and self.claude_api_key:
#                 # Use selected model if available, otherwise default to Claude 3.5 Sonnet
#                 model_name = self.selected_model if self.selected_model else 'claude-3-5-sonnet-20241022'
#                 self.translation_ai_provider = AIProviderFactory.create_provider(
#                     'claude', self.claude_api_key, self.websocket, model_name=model_name
#                 )
#             elif self.translation_provider == 'gemini' and self.gemini_api_key:
#                 # Use selected model if available, otherwise default to latest pro model
#                 model_name = self.selected_model if self.selected_model else 'gemini-1.5-pro-latest'
#                 self.translation_ai_provider = AIProviderFactory.create_provider(
#                     'gemini', self.gemini_api_key, self.websocket, model_name=model_name
#                 )
#             else:
#                 pass
            


#             # Initialize AI features provider (for summarization and keywords)
#             if self.ai_provider == 'openai' and self.openai_api_key:
#                 self.ai_features_provider = AIProviderFactory.create_provider(
#                     'openai', self.openai_api_key, self.websocket, model_name=self.selected_model
#                 )
#             elif self.ai_provider == 'claude' and self.claude_api_key:
#                 # Use selected model if available, otherwise default to Claude 3.5 Sonnet
#                 model_name = self.selected_model if self.selected_model else 'claude-3-5-sonnet-20241022'
#                 self.ai_features_provider = AIProviderFactory.create_provider(
#                     'claude', self.claude_api_key, self.websocket, model_name=model_name
#                 )
#             elif self.ai_provider == 'gemini' and self.gemini_api_key:
#                 # Use selected model if available, otherwise default to latest pro model
#                 model_name = self.selected_model if self.selected_model else 'gemini-1.5-pro-latest'
#                 self.ai_features_provider = AIProviderFactory.create_provider(
#                     'gemini', self.gemini_api_key, self.websocket, model_name=model_name
#                 )
#             else:
#                 pass
            
            
#         except Exception as e:
#             self.translation_ai_provider = None
#             self.ai_features_provider = None

#     async def get_summarizer(self, text):
#         '''Uses the AI provider to get the summary from the text'''
#         if not self.ai_features_provider:
#             return None
            
#         try:
#             return await self.ai_features_provider.summarize(text, self.system_prompt_summarizer)
#         except Exception as e:
#             return None

#     async def get_keywords(self, text):
#         '''Uses the AI provider to get the keywords from the text'''
#         if not self.ai_features_provider:
#             return {"keywords": [], "error": "No AI provider available"}
            
#         try:
#             return await self.ai_features_provider.extract_keywords(text, self.system_prompt_keywords)
#         except Exception as e:
#             return {"keywords": [], "error": f"Keyword extraction error: {str(e)}"}

#     def process_speaker_diarization(self, result):
#         """Process speaker diarization information from Deepgram result."""
#         try:
#             transcript = result.channel.alternatives[0].transcript
            
#             if not result.channel.alternatives[0].words:
#                 speaker_id = 0
#                 start_time = 0.0
#                 end_time = 0.0
#             else:
#                 first_word = result.channel.alternatives[0].words[0]
#                 speaker_id = getattr(first_word, 'speaker', 0)  # Default to speaker 0 if no speaker info
#                 start_time = first_word.start
                
#                 # Get the last word for end time
#                 last_word = result.channel.alternatives[0].words[-1]
#                 end_time = last_word.end
            
            
#             # Create speaker segment
#             speaker_segment = {
#                 'speaker_id': speaker_id,
#                 'start_time': start_time,
#                 'end_time': end_time,
#                 'transcript': transcript,
#                 'is_final': result.is_final
#             }
#             print(f"Speaker Segment: {speaker_segment}")

#             # Initialize speaker transcript if not exists
#             if speaker_id not in self.speaker_transcript:
#                 self.speaker_transcript[speaker_id] = ""
            
#             # Add to speaker segments and update speaker transcript
#             if result.is_final:
#                 self.speaker_segments.append(speaker_segment)
#                 # Clean up transcript before adding
#                 clean_transcript = transcript.strip()
#                 if clean_transcript:
#                     self.speaker_transcript[speaker_id] += f" {clean_transcript}"
                
#             return speaker_segment
            
#         except Exception as e:
#             return None

#     async def handle_generic_translation_error(self, error_message):
#         """Handle generic translation errors."""
#         try:
#             await self.websocket.send_json({
#                 'type': 'error',
#                 'content': f'Translation service error: {error_message}',
#                 'error_code': 'TRANSLATION_ERROR'
#             })
#         except Exception as e:
#             pass

#     async def handle_openai_error(self, status_code: int, response_text: str):
#         """Handle OpenAI API errors and show them on the side panel."""
#         try:
#             error_data = json.loads(response_text) if response_text else {}
#             error_message = error_data.get('error', {}).get('message', 'Unknown OpenAI API error')
#             error_type = error_data.get('error', {}).get('type', 'unknown_error')
            
#             # Handle different types of OpenAI errors
#             if status_code == 429:
#                 if 'quota' in error_message.lower() or 'insufficient_quota' in error_type:
#                     user_message = "Your OpenAI API quota has been exceeded. Please check your OpenAI billing and add credits to continue using translation features."
#                     error_code = 'OPENAI_QUOTA_EXCEEDED'
#                 elif 'rate_limit' in error_message.lower():
#                     user_message = "OpenAI API rate limit exceeded. Please wait a moment and try again."
#                     error_code = 'OPENAI_RATE_LIMIT'
#                 else:
#                     user_message = f"OpenAI API error: {error_message}"
#                     error_code = 'OPENAI_ERROR_429'
#             elif status_code == 401:
#                 user_message = "OpenAI API key is invalid or unauthorized. Please check your API key configuration."
#                 error_code = 'OPENAI_UNAUTHORIZED'
#             elif status_code == 403:
#                 user_message = "OpenAI API access forbidden. Please verify your API key permissions."
#                 error_code = 'OPENAI_FORBIDDEN'
#             else:
#                 user_message = f"OpenAI API error ({status_code}): {error_message}"
#                 error_code = 'OPENAI_API_ERROR'
            
#             # Send error to frontend to display on side panel
#             try:
#                 await self.websocket.send_json({
#                     'type': 'translation_error',
#                     'content': user_message,
#                     'error_code': error_code,
#                     'status_code': status_code
#                 })
#             except Exception as send_error:
#                 pass
                
#         except Exception as e:
#             # Fallback error message
#             try:
#                 await self.websocket.send_json({
#                     'type': 'translation_error',
#                     'content': f'OpenAI translation service error (Code: {status_code}). Please check your API configuration.',
#                     'error_code': 'OPENAI_ERROR_FALLBACK'
#                 })
#             except:
#                 pass  # If WebSocket is closed, we can't send anything

#     async def connect_to_openai(self):
#         try:
#             """Establish connection to OpenAI's realtime API."""
#             # Use additional_headers for newer websockets versions
#             try:
#                 self.openai_ws = await websockets.connect(
#                     'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview',
#                     additional_headers={
#                         "Authorization": f"Bearer {self.openai_api_key}",
#                         "OpenAI-Beta": "realtime=v1"
#                     }
#                 )
#             except TypeError:
#                 # Fallback for older websockets versions
#                 self.openai_ws = await websockets.connect(
#                     'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview',
#                     extra_headers={
#                         "Authorization": f"Bearer {self.openai_api_key}",
#                         "OpenAI-Beta": "realtime=v1"
#                     }
#                 )
            
#             # Initialize session
#             session_update = {
#                 "type": "session.update",
#                 "session": {
#                     "instructions": self.system_prompt,
#                     "modalities": ["text"],
#                     "temperature": 0.6,
#                 }
#             }
#             await self.openai_ws.send(json.dumps(session_update))

#         except Exception as e:
#             print(f"Error connecting to OpenAI: {e}")
#             raise e


#     async def process_openai_responses(self):
#         """Process responses from OpenAI's realtime API."""
#         try:
#             if self.openai_ws is None:
#                 print("OpenAI WebSocket is not connected")
#                 return
                
#             async for message in self.openai_ws:
#                 try:
#                     response = json.loads(message)
#                     if response.get('type') == 'response.text.delta':
#                         # Use safe websocket send to avoid ASGI errors when websocket is closed
#                         success = await self._safe_websocket_send({
#                             'type': 'assistant',
#                             'content': response.get('delta'),
#                         })
#                         if not success:
#                             return

#                     elif response.get('type') == 'response.text.done':
#                         # Use safe websocket send to avoid ASGI errors when websocket is closed
#                         success = await self._safe_websocket_send({
#                             'type': 'assistant_done',
#                             'content': 'Completed',
#                         })
#                         if not success:
#                             return
                        
#                 except json.JSONDecodeError as e:
#                     print(f"Error parsing OpenAI response: {e}")
#                 except Exception as e:
#                     print(f"Error processing OpenAI message: {e}")
                
#         except websockets.exceptions.ConnectionClosed:
#             print("OpenAI connection closed")
#             raise Exception('OpenAI connection closed')
#         except Exception as e:
#             print(f"Error processing OpenAI responses: {e}")
#             raise e

#     async def process_gemini_responses(self):
#         """Process responses from Gemini's realtime API."""

#         print(f"Starting Gemini Live API response processing...")
#         try:
#             if self.gemini_ws is None:
#                 print("Gemini WebSocket is not connected")
#                 return

#             print(f"Gemini WebSocket connected, waiting for messages...")

#             async for message in self.gemini_ws:
#                 try:
#                     response = json.loads(message)
#                     print(f"Received Gemini Live API message: {response}")

#                     # Handle Gemini Live API response format
#                     # According to https://ai.google.dev/api/live#websocket-connection
                    
#                     # Check for content in response (text generation)
#                     if "content" in response:
#                         content = response["content"]
#                         if "parts" in content:
#                             for part in content["parts"]:
#                                 success = await self._process_text_part(part)
#                                 if not success:
#                                     return
                    
#                     # Check for candidates in response (alternative format)
#                     elif "candidates" in response:
#                         candidates = response["candidates"]
#                         if candidates and "content" in candidates[0]:
#                             content = candidates[0]["content"]
#                             if "parts" in content:
#                                 for part in content["parts"]:
#                                     success = await self._process_text_part(part)
#                                     if not success:
#                                         return
                    
#                     # End of stream - check for finish reason or usage metadata
#                     if response.get("finishReason") is not None or "usageMetadata" in response:
#                         print("Gemini Live API response complete")
#                         success = await self._safe_websocket_send({
#                             'type': 'assistant_done',
#                             'content': 'Completed',
#                         })
#                         if not success:
#                             return
#                 except json.JSONDecodeError as e:
#                     print(f"Error parsing Gemini response: {e}")
#                 except Exception as e:
#                     print(f"Error processing Gemini message: {e}")
#         except websockets.exceptions.ConnectionClosed:
#             print("Gemini connection closed")
#             print(f"Gemini WebSocket: {self.gemini_ws}")
#             raise Exception('Gemini connection closed')
#         except Exception as e:
#             print(f"Error processing Gemini responses: {e}")
#             raise e

#     async def _safe_websocket_send(self, message: dict):
#         """Safely send a message to WebSocket without raising exceptions if already closed."""
#         try:
#             # Check websocket state like the original code did
#             if self.websocket.client_state != WebSocketState.DISCONNECTED:
#                 await self.websocket.send_json(message)
#                 return True
#         except Exception as e:
#             # WebSocket is closed or error occurred - silently handle like original
#             return False
#         return False

#     async def _process_text_part(self, part):
#         """Helper to process a text part from Gemini response."""
#         if "text" in part:
#             text_content = part["text"]
#             if text_content:
#                 clean_content = text_content.replace('*', '').replace('#', '').strip()
#                 if clean_content: # Ensure content isn't empty after cleaning
#                     success = await self._safe_websocket_send({
#                         'type': 'assistant',
#                         'content': clean_content,
#                     })
#                     return success
#         return True

#     async def send_message_to_openai(self, text):
#         """Send a message to OpenAI's realtime API."""
#         try:
#             if self.openai_ws is None:
#                 print("OpenAI WebSocket is not connected")
#                 return
                
#             conversation_item = {
#                 "type": "conversation.item.create",
#                 "item": {
#                     "type": "message",
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "input_text",
#                             "text": text
#                         }
#                     ]
#                 }
#             }
#             await self.openai_ws.send(json.dumps(conversation_item))
#             await self.openai_ws.send(json.dumps({"type": "response.create"}))
        
#         except Exception as e:
#             print(f"Error sending to OpenAI: {e}")

#     async def translate_with_error_handling(self, text, timestamp=None):
#         """Wrapper for translate_text with proper error handling that doesn't break transcription."""
#         # Skip translation if it's been disabled due to repeated failures
#         if not self.translation_enabled:
#             return
            

#         try:
#             await self.translate_text(text, timestamp)
#             # Reset failure counter on successful translation
#             self.consecutive_translation_failures = 0
            
#         except Exception as translation_error:
#             self.consecutive_translation_failures += 1
            
#             # Determine error type and appropriate response
#             error_type = self._classify_translation_error(str(translation_error))
            
#             # Send error to frontend but don't break transcription
#             try:
#                 if error_type == 'QUOTA_EXCEEDED':
#                     await self._handle_quota_exceeded_error(text, str(translation_error), timestamp)
#                 elif error_type == 'AUTH_ERROR':
#                     await self._handle_auth_error(text, str(translation_error), timestamp)
#                 else:
#                     await self._handle_generic_translation_failure(text, str(translation_error), timestamp)
                    
#             except Exception as websocket_error:
#                 pass
                
#             # Disable translation after too many consecutive failures
#             if self.consecutive_translation_failures >= self.max_translation_failures:
#                 await self._disable_translation_mode()
    
#     def _classify_translation_error(self, error_message: str) -> str:
#         """Classify translation error to determine appropriate handling."""
#         error_lower = error_message.lower()
        
#         if any(keyword in error_lower for keyword in ['quota', 'credits', 'billing', 'payment', 'insufficient']):
#             return 'QUOTA_EXCEEDED'
#         elif any(keyword in error_lower for keyword in ['unauthorized', 'auth', 'api key', 'forbidden']):
#             return 'AUTH_ERROR'
#         elif any(keyword in error_lower for keyword in ['rate limit', 'too many requests']):
#             return 'RATE_LIMIT'
#         else:
#             return 'GENERIC_ERROR'
    
#     async def _handle_quota_exceeded_error(self, text: str, error_message: str, timestamp=None):
#         """Handle quota exceeded errors with prominent notification."""
#         await self._safe_websocket_send({
#             'type': 'translation_error',
#             'content': f'Translation service out of credits. Transcription will continue but translations are disabled until credits are added.',
#             'error_code': 'TRANSLATION_QUOTA_EXCEEDED',
#             'severity': 'critical',
#             'persistent': True,  # Frontend should show persistent notification
#             'original_text': text[:30] + "..." if len(text) > 30 else text,
#             'timestamp': timestamp
#         })
        
#     async def _handle_auth_error(self, text: str, error_message: str, timestamp=None):
#         """Handle authentication errors."""
#         await self._safe_websocket_send({
#             'type': 'translation_error', 
#             'content': f'Translation service authentication failed. Please check your API keys in settings.',
#             'error_code': 'TRANSLATION_AUTH_ERROR',
#             'severity': 'critical',
#             'persistent': True,
#             'original_text': text[:30] + "..." if len(text) > 30 else text,
#             'timestamp': timestamp
#         })
        
#     async def _handle_generic_translation_failure(self, text: str, error_message: str, timestamp=None):
#         """Handle generic translation failures."""
#         await self._safe_websocket_send({
#             'type': 'translation_error',
#             'content': f'Translation temporarily unavailable. Transcription continues normally.',
#             'error_code': 'TRANSLATION_TEMPORARY_ERROR', 
#             'severity': 'warning',
#             'persistent': False,
#             'original_text': text[:30] + "..." if len(text) > 30 else text,
#             'timestamp': timestamp
#         })
        
#     async def _disable_translation_mode(self):
#         """Disable translation after repeated failures."""
#         self.translation_enabled = False
        
#         await self.websocket.send_json({
#             'type': 'translation_disabled',
#             'content': f'Translation temporarily disabled due to repeated failures. Transcription will continue normally.',
#             'error_code': 'TRANSLATION_DISABLED',
#             'severity': 'warning',
#             'can_retry': True  # Frontend can offer retry option
#         })
    
#     async def retry_translation(self):
#         """Re-enable translation after user request."""
#         self.translation_enabled = True
#         self.consecutive_translation_failures = 0
        
#         await self.websocket.send_json({
#             'type': 'translation_enabled',
#             'content': 'Translation has been re-enabled. New transcripts will be translated.',
#             'error_code': 'TRANSLATION_ENABLED'
#         })
    
#     async def _handle_frontend_command(self, command: dict):
#         """Handle commands received from the frontend via WebSocket."""
#         command_type = command.get('type')
        
#         try:
#             if command_type == 'retry_translation':
#                 await self.retry_translation()
                
#             elif command_type == 'ping':
#                 # Respond to ping with pong and update activity
#                 self.update_activity()
#                 await self._safe_websocket_send({
#                     'type': 'pong',
#                     'timestamp': command.get('timestamp'),
#                     'server_time': datetime.now().isoformat()
#                 })
                
#             elif command_type == 'request_status':
#                 # Send current status to frontend
#                 await self._safe_websocket_send({
#                     'type': 'status_update',
#                     'translation_enabled': self.translation_enabled,
#                     'consecutive_failures': self.consecutive_translation_failures,
#                     'transcription_active': not self.finish_event.is_set()
#                 })
                
#             else:
#                 pass
#         except Exception as e:
#             pass

#     async def translate_text(self, text, timestamp=None):
#         """Route translation to the appropriate provider using polymorphism with retry logic for real-time sentence translation."""
#         # Skip translation for very short or empty text
#         if not text or len(text.strip()) < 3:
#             return
            
#         max_retries = 2  # Reduced retries for faster response
#         retry_delay = 0.5  # Faster retry for real-time experience
        
#         for attempt in range(max_retries):
#             try:
#                 # Update activity when starting translation
#                 self.update_activity()
                
                
#                 # Handle different translation providers and models
#                 if self.translation_provider == 'openai' and self.selected_model == 'gpt-4o-realtime-preview':
#                     # OpenAI Realtime API for streaming translation
#                     await self.send_message_to_openai(text)
                
#                 elif self.translation_provider in ['claude', 'gemini', 'openai']:
#                     # Use streaming for Claude, Gemini, and OpenAI REST API
#                     if not self.translation_ai_provider:
#                         await self.handle_generic_translation_error(f"{self.translation_provider.title()} translation provider not configured")
#                         return
                    
#                     # Special handling for Gemini WebSocket
#                     if self.translation_provider == 'gemini' and hasattr(self.translation_ai_provider, 'gemini_ws') and self.translation_ai_provider.gemini_ws:
#                         # Use Gemini WebSocket for real-time translation
#                         try:
#                             await self.translation_ai_provider.send_translation_request(text, self.system_prompt)
#                             # The response will be handled by process_gemini_responses
#                             return
#                         except Exception as e:
#                             print(f"Gemini WebSocket failed, falling back to REST API: {e}")
#                             # Fall back to REST API if WebSocket fails
#                             if hasattr(self.translation_ai_provider, 'supports_streaming') and self.translation_ai_provider.supports_streaming():
#                                 await self.translation_ai_provider.translate_streaming(
#                                     text, self.target_language, self.system_prompt, timestamp
#                                 )
#                                 return
#                             else:
#                                 await self.handle_generic_translation_error(f"Gemini translation failed: {str(e)}")
#                                 return
                    
#                     # Check if provider supports streaming
#                     if hasattr(self.translation_ai_provider, 'supports_streaming') and self.translation_ai_provider.supports_streaming():
#                         # Use streaming translation
#                         await self.translation_ai_provider.translate_streaming(
#                             text, self.target_language, self.system_prompt, timestamp
#                         )
#                     else:
                        
#                         # Send translation start signal
#                         try:
#                             await self.websocket.send_json({
#                                 'type': 'translation_start',
#                                 'content': f'Translating with {self.translation_provider.title()} ({self.selected_model}): {text[:30]}...' if len(text) > 30 else f'Translating with {self.translation_provider.title()}: {text}',
#                                 'timestamp': timestamp
#                             })
#                         except WebSocketDisconnect:
#                             self.finish_event.set()
#                             return
#                         except Exception as e:
#                             pass
                        
#                         translation = await self.translation_ai_provider.translate(
#                             text, self.target_language, self.system_prompt
#                         )
                        
#                         if translation:
#                             # Store translation for session summary
#                             self.Complete_Translation += f" {translation}"
                            
#                             # Send translated text to client
#                             try:
#                                 await self.websocket.send_json({
#                                     'type': 'translation',
#                                     'content': translation,
#                                     'timestamp': timestamp
#                                 })
#                             except WebSocketDisconnect:
#                                 self.finish_event.set()
#                                 return
#                             except Exception as e:
#                                 pass
#                         else:
#                             await self.handle_generic_translation_error(f"Empty translation received from {self.translation_provider.title()}")
#                             return
#                 else:
#                     # Fallback for unknown providers
#                     await self.handle_generic_translation_error(f"Unknown translation provider: {self.translation_provider}")
#                     return
                
#                 return
                
#             except httpx.TimeoutException:
#                 if attempt < max_retries - 1:
#                     await asyncio.sleep(retry_delay)
#                     continue
#                 else:
#                     try:
#                         await self.websocket.send_json({
#                             'type': 'translation_error',
#                             'content': 'Translation service temporarily slow. Continuing with next sentences.',
#                             'error_code': 'SENTENCE_TIMEOUT_ERROR'
#                         })
#                     except:
#                         pass
#                     return
                        
#             except WebSocketDisconnect as e:
#                 self.finish_event.set()
#                 return
                
#             except Exception as e:
#                 if attempt < max_retries - 1:
#                     await asyncio.sleep(retry_delay)
#                     continue
#                 else:
                    
#                     try:
#                         await self.websocket.send_json({
#                             'type': 'translation_error',
#                             'content': f'Failed to translate sentence. Continuing with next sentences.',
#                             'error_code': 'SENTENCE_TRANSLATION_ERROR'
#                         })
#                     except:
#                         pass
#                     return

#     async def Premium_Feature(self):
#         self.Object_session["Keywords"] = await self.get_keywords(self.Complete_Translation)
#         self.Object_session["Summary"] = await self.get_summarizer(self.Complete_Translation) 

#     async def _reconnect_deepgram(self, max_retries=3, retry_delay=2.0):
#         """Attempt to reconnect to Deepgram with exponential backoff."""
#         for attempt in range(max_retries):
#             try:
#                 print(f"Attempting Deepgram reconnection (attempt {attempt + 1}/{max_retries})")
                
#                 # Send notification to client about reconnection attempt
#                 try:
#                     await self._safe_websocket_send({
#                         'type': 'info',
#                         'content': f'Reconnecting to transcription service... (attempt {attempt + 1})',
#                         'error_code': 'DEEPGRAM_RECONNECTING'
#                     })
#                 except:
#                     pass
                
#                 # Reset connection flag
#                 self.deepgram_connection_lost = False
                
#                 # Create new connection
#                 try:
#                     dg_connection = self.deepgram.listen.asyncwebsocket.v('1')
#                 except AttributeError:
#                     dg_connection = self.deepgram.listen.asynclive.v('1')
                
#                 # Set up event handlers again
#                 assistant_instance = self
                
#                 async def on_message(self_handler, result, **kwargs):
#                     sentence = result.channel.alternatives[0].transcript
                    
#                     if len(sentence) == 0:
#                         return
                    
#                     # Process speaker diarization
#                     speaker_segment = self.process_speaker_diarization(result)
                    
#                     # Get start time safely
#                     start_time = 0.0
#                     if result.channel.alternatives[0].words:
#                         start_time = result.channel.alternatives[0].words[0].start
#                         if self.stime == 0:
#                             self.stime = start_time
                    
#                     # Update last transcript timestamp
#                     self.last_transcript_timestamp = start_time
                    
#                     print(f"Sentence received: {sentence} (is_final: {result.is_final})")
                    
#                     if result.is_final:
#                         self.transcript_parts.append(sentence)
#                         self.Complete_Transcript += f" {sentence}"
                        
#                         transcript_data = {
#                             'type': 'transcript_final', 
#                             'content': sentence, 
#                             'time': float(start_time),
#                             'speaker_info': speaker_segment
#                         }
                        
#                         if self.is_paid_user() and self.user_credits is not None:
#                             transcript_data["User_Credits"] = self.user_credits
                        
#                         try:
#                             # Send transcript to frontend immediately
#                             await self._safe_websocket_send(transcript_data)
#                             print(f"Sent transcript to frontend: {sentence}")
                            
#                             # Trigger translation immediately for final transcripts
#                             if self.translation_enabled and sentence.strip():
#                                 print(f"Triggering translation for: {sentence}")
#                                 translation_task = asyncio.create_task(
#                                     self.translate_with_error_handling(sentence, start_time)
#                                 )
#                                 translation_task.add_done_callback(self._handle_translation_task_completion)
#                         except WebSocketDisconnect:
#                             print("WebSocket disconnected during transcript send")
#                             self.finish_event.set()
#                             return
#                         except Exception as e:
#                             print(f"Error sending transcript: {e}")
                    
#                     else:
#                         # Handle interim results
#                         transcript_data = {
#                             'type': 'transcript_interim', 
#                             'content': sentence, 
#                             'time': float(start_time) if start_time else 0,
#                             'speaker_info': speaker_segment
#                         }
                        
#                         if self.is_paid_user() and self.user_credits is not None:
#                             transcript_data["User_Credits"] = self.user_credits
                        
#                         try:
#                             await self._safe_websocket_send(transcript_data)
#                         except WebSocketDisconnect:
#                             print("WebSocket disconnected during interim transcript send")
#                             self.finish_event.set()
#                             return
#                         except Exception as e:
#                             print(f"Error sending interim transcript: {e}")

#                 async def on_metadata(self_handler, metadata, **kwargs):
#                     print(f"Metadata: {metadata}")

#                 async def on_speech_started(self_handler, speech_started, **kwargs):
#                     print(f"Speech Started")

#                 async def on_utterance_end(self_handler, utterance_end, **kwargs):
#                     if len(self.transcript_parts) > 0:
#                         recent_transcript = self.transcript_parts[-1] if self.transcript_parts else ""
#                         if recent_transcript.strip() and len(recent_transcript.strip()) > 5:
#                             task = asyncio.create_task(self.translate_with_error_handling(recent_transcript, self.last_transcript_timestamp))
#                             task.add_done_callback(self._handle_translation_task_completion)

#                 async def on_close(self_handler, close, **kwargs):
#                     print(f"Deepgram connection closed")
#                     assistant_instance.deepgram_connection_lost = True

#                 async def on_error(self_handler, error, **kwargs):
#                     print(f"Deepgram error: {error}")
#                     error_message = str(error).lower()
                    
#                     if "1011" in error_message or "timeout" in error_message or "net0001" in error_message:
#                         print("Deepgram timeout detected during reconnected session")
#                         assistant_instance.deepgram_connection_lost = True
#                         # Explicitly close the Deepgram connection immediately
#                         try:
#                             await self_handler.finish()
#                         except Exception as e_close:
#                             print(f"Error closing Deepgram connection in on_error (reconnect): {e_close}")
#                         # Signal end of session
#                         assistant_instance.finish_event.set()
                        
#                         # Send session ended message to frontend
#                         try:
#                             await assistant_instance._safe_websocket_send({
#                                 'type': 'session_ended_no_audio',
#                                 'content': 'Session ended due video not sending any audio data.',
#                                 'error_code': 'DEEPGRAM_SESSION_ENDED'
#                             })
#                         except Exception as send_error:
#                             print(f"Error sending session ended message: {send_error}")
#                         return
                    
#                     # Handle other errors
#                     if "insufficient credits" in error_message or "quota exceeded" in error_message:
#                         try:
#                             await assistant_instance._safe_websocket_send({
#                                 'type': 'error',
#                                 'content': 'Deepgram API credits exhausted.',
#                                 'error_code': 'DEEPGRAM_CREDITS_EXHAUSTED'
#                             })
#                         except:
#                             pass
#                     elif "unauthorized" in error_message or "invalid api key" in error_message:
#                         try:
#                             await assistant_instance._safe_websocket_send({
#                                 'type': 'error',
#                                 'content': 'Deepgram API key is invalid.',
#                                 'error_code': 'DEEPGRAM_AUTH_ERROR'
#                             })
#                         except:
#                             pass

#                 async def on_open(self_handler, open, **kwargs):
#                     print(f"Deepgram reconnection successful")

#                 async def on_unhandled(self_handler, unhandled, **kwargs):
#                     print(f"Unhandled Deepgram message: {unhandled}")

#                 # Register handlers
#                 dg_connection.on(LiveTranscriptionEvents.Open, on_open)
#                 dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
#                 dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
#                 dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)
#                 dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
#                 dg_connection.on(LiveTranscriptionEvents.Close, on_close)
#                 dg_connection.on(LiveTranscriptionEvents.Error, on_error)
#                 dg_connection.on(LiveTranscriptionEvents.Unhandled, on_unhandled)

#                 # Start the connection
#                 connection_result = await dg_connection.start(self.dg_connection_options)
                
#                 if connection_result is False:
#                     raise Exception('Failed to reconnect to Deepgram')
                
#                 # Success! Reset connection flag and restart keep-alive
#                 self.deepgram_connection_lost = False
                
#                 # Cancel old keep-alive task if it exists
#                 if hasattr(self, 'deepgram_keepalive_task') and self.deepgram_keepalive_task:
#                     self.deepgram_keepalive_task.cancel()
#                     try:
#                         await self.deepgram_keepalive_task
#                     except asyncio.CancelledError:
#                         pass
                
#                 # Start new keep-alive task
#                 self.deepgram_keepalive_task = asyncio.create_task(self._deepgram_keepalive(dg_connection))
                
#                 # Update last audio time
#                 self.last_audio_sent_time = datetime.now()
                
#                 # Notify client of successful reconnection
#                 try:
#                     await self._safe_websocket_send({
#                         'type': 'info',
#                         'content': 'Transcription service reconnected successfully!',
#                         'error_code': 'DEEPGRAM_RECONNECTED'
#                     })
#                 except:
#                     pass
                
#                 print(f"Deepgram reconnection successful on attempt {attempt + 1}")
#                 return dg_connection
                
#             except Exception as reconnect_error:
#                 print(f"Reconnection attempt {attempt + 1} failed: {reconnect_error}")
                
#                 if attempt < max_retries - 1:
#                     # Exponential backoff
#                     wait_time = retry_delay * (2 ** attempt)
#                     print(f"Waiting {wait_time} seconds before next attempt...")
#                     await asyncio.sleep(wait_time)
#                 else:
#                     # All attempts failed
#                     try:
#                         await self._safe_websocket_send({
#                             'type': 'error',
#                             'content': 'Failed to reconnect to transcription service after multiple attempts. Please refresh the page.',
#                             'error_code': 'DEEPGRAM_RECONNECTION_FAILED'
#                         })
#                     except:
#                         pass
#                     raise Exception(f'Failed to reconnect to Deepgram after {max_retries} attempts')
        
#         return None

#     async def _wait_for_audio_or_finish(self):
#         """Wait for audio to be sent from frontend before attempting reconnection"""
#         try:
#             print("Waiting for audio to resume before reconnecting...")
#             # Reset the audio flag to ensure clean state
#             self.audio_being_sent = False
            
#             while not self.finish_event.is_set() and not self.audio_being_sent:
#                 # Listen for WebSocket messages without a Deepgram connection
#                 try:
#                     if self.websocket.client_state == WebSocketState.DISCONNECTED:
#                         self.finish_event.set()
#                         break
                    
#                     # Use a short timeout to periodically check conditions
#                     message = await asyncio.wait_for(self.websocket.receive(), timeout=2.0)
                    
#                     if message["type"] == "websocket.receive":
#                         if "bytes" in message and len(message["bytes"]) > 0:
#                             # Audio data detected - set flag and update timestamp
#                             print("Audio data detected - ready to reconnect")
#                             self.audio_being_sent = True
#                             self.last_audio_sent_time = datetime.now()
#                             # Store the audio data temporarily so it's not lost
#                             if not hasattr(self, '_pending_audio_data'):
#                                 self._pending_audio_data = []
#                             self._pending_audio_data.append(message["bytes"])
#                             # Limit buffer size to prevent memory issues
#                             if len(self._pending_audio_data) > 50:  # Keep last 50 chunks
#                                 self._pending_audio_data = self._pending_audio_data[-50:]
#                             break
#                         elif "text" in message:
#                             # Handle frontend commands even while waiting
#                             try:
#                                 command = json.loads(message["text"])
#                                 await self._handle_frontend_command(command)
#                                 # Check if command was to stop/finish session
#                                 if command.get('action') == 'stop_session' or command.get('action') == 'end_session':
#                                     self.finish_event.set()
#                                     break
#                             except:
#                                 pass
#                     elif message["type"] == "websocket.disconnect":
#                         self.finish_event.set()
#                         break
                        
#                 except asyncio.TimeoutError:
#                     # Timeout is expected - continue waiting and check conditions
#                     print("Still waiting for audio data to resume...")
#                     continue
#                 except WebSocketDisconnect:
#                     self.finish_event.set()
#                     break
#                 except Exception as e:
#                     print(f"Error while waiting for audio: {e}")
#                     # Continue waiting unless it's a critical error
#                     if "disconnect" in str(e).lower():
#                         self.finish_event.set()
#                         break
#                     continue
                    
#         except Exception as e:
#             print(f"Error in _wait_for_audio_or_finish: {e}")
#             # Set finish event on critical errors
#             self.finish_event.set()

#     async def transcribe_audio(self):
#         if self.deepgram is None:
#             return
            
#         # Main loop with reconnection logic
#         while not self.finish_event.is_set():
#             try:
#                 # Check if we need to reconnect immediately
#                 if self.deepgram_connection_lost or self.pending_reconnection:
#                     # Check if audio is currently being sent
#                     if self.audio_being_sent:
#                         print("Audio is being sent - attempting immediate reconnection...")
#                         # Directly call the reconnection
#                         self.pending_reconnection = False
#                         await self._transcribe_audio_session()
#                         continue
#                     else:
#                         print("No audio being sent - waiting for audio to resume before reconnecting...")
#                         self.pending_reconnection = True
#                         # Wait for audio to be sent before reconnecting
#                         await self._wait_for_audio_or_finish()
#                         if not self.finish_event.is_set() and self.audio_being_sent:
#                             print("Audio resumed - attempting reconnection...")
#                             self.pending_reconnection = False
#                             continue
#                         elif self.finish_event.is_set():
#                             break
#                         else:
#                             # Still no audio, continue waiting
#                             continue
#                 else:
#                     # Normal transcription session
#                     await self._transcribe_audio_session()
                    
#             except Exception as e:
#                 print(f"Transcription session error: {e}")
#                 error_msg = str(e).lower()
                
#                 # Check if this is a connection loss error
#                 if ("1011" in error_msg or "timeout" in error_msg or 
#                     "connection" in error_msg or "reconnection failed" in error_msg):
#                     print("Connection lost - marking for reconnection...")
#                     self.deepgram_connection_lost = True
#                     self.audio_being_sent = False
#                     # Continue to reconnection logic
#                     continue
#                 else:
#                     # Non-reconnection error, end session
#                     print(f"Non-recoverable error: {e}")
#                     break

#     async def _transcribe_audio_session(self):
#         """Single transcription session that can be restarted on connection loss"""
#         # Capture the Assistant instance for use in callbacks
#         assistant_instance = self

#         async def on_message(self_handler, result, **kwargs):
#             sentence = result.channel.alternatives[0].transcript
            
#             if len(sentence) == 0:
#                 return
            
#             # Process speaker diarization
#             speaker_segment = self.process_speaker_diarization(result)
            
#             # Get start time safely
#             start_time = 0.0
#             if result.channel.alternatives[0].words:
#                 start_time = result.channel.alternatives[0].words[0].start
#                 if self.stime == 0:
#                     self.stime = start_time
            
#             # Update last transcript timestamp
#             self.last_transcript_timestamp = start_time
            
#             print(f"Sentence received: {sentence} (is_final: {result.is_final})")
            
#             if result.is_final:
#                 self.transcript_parts.append(sentence)
#                 self.Complete_Transcript += f" {sentence}"
                
#                 transcript_data = {
#                     'type': 'transcript_final', 
#                     'content': sentence, 
#                     'time': float(start_time),
#                     'speaker_info': speaker_segment
#                 }
                
#                 if self.is_paid_user() and self.user_credits is not None:
#                     transcript_data["User_Credits"] = self.user_credits
                
#                 try:
#                     # Send transcript to frontend immediately
#                     await self._safe_websocket_send(transcript_data)
#                     print(f"Sent transcript to frontend: {sentence}")
                    
#                     # Trigger translation immediately for final transcripts
#                     if self.translation_enabled and sentence.strip():
#                         print(f"Triggering translation for: {sentence}")
#                         translation_task = asyncio.create_task(
#                             self.translate_with_error_handling(sentence, start_time)
#                         )
#                         translation_task.add_done_callback(self._handle_translation_task_completion)
#                 except WebSocketDisconnect:
#                     print("WebSocket disconnected during transcript send")
#                     self.finish_event.set()
#                     return
#                 except Exception as e:
#                     print(f"Error sending transcript: {e}")
            
#             else:
#                 # Handle interim results
#                 transcript_data = {
#                     'type': 'transcript_interim', 
#                     'content': sentence, 
#                     'time': float(start_time) if start_time else 0,
#                     'speaker_info': speaker_segment
#                 }
                
#                 if self.is_paid_user() and self.user_credits is not None:
#                     transcript_data["User_Credits"] = self.user_credits
                
#                 try:
#                     await self._safe_websocket_send(transcript_data)
#                 except WebSocketDisconnect:
#                     print("WebSocket disconnected during interim transcript send")
#                     self.finish_event.set()
#                     return
#                 except Exception as e:
#                     print(f"Error sending interim transcript: {e}")

#         async def on_metadata(self_handler, metadata, **kwargs):
#             print(f"Metadata: {metadata}")

#         async def on_speech_started(self_handler, speech_started, **kwargs):
#             print(f"Speech Started")

#         async def on_utterance_end(self_handler, utterance_end, **kwargs):
#             if len(self.transcript_parts) > 0:
#                 recent_transcript = self.transcript_parts[-1] if self.transcript_parts else ""
#                 if recent_transcript.strip() and len(recent_transcript.strip()) > 5:
#                     task = asyncio.create_task(self.translate_with_error_handling(recent_transcript, self.last_transcript_timestamp))
#                     task.add_done_callback(self._handle_translation_task_completion)

#         async def on_close(self_handler, close, **kwargs):
#             print(f"Deepgram connection closed")
#             assistant_instance.deepgram_connection_lost = True
#             # Reset audio flag since connection is closed
#             assistant_instance.audio_being_sent = False

#         async def on_error(self_handler, error, **kwargs):
#             print(f"Deepgram error: {error}")
#             error_message = str(error).lower()
            
#             # Check for connection timeout errors (error 1011)
#             if "1011" in error_message or "timeout" in error_message or "net0001" in error_message:
#                 print("Deepgram timeout detected - marking for reconnection")
#                 assistant_instance.deepgram_connection_lost = True
#                 # Reset audio flag since connection is lost
#                 assistant_instance.audio_being_sent = False
                
#                 # Explicitly close the Deepgram connection immediately
#                 try:
#                     await self_handler.finish()
#                 except Exception as e_close:
#                     print(f"Error closing Deepgram connection in on_error: {e_close}")

#                 # Try to send warning to client about reconnection
#                 try:
#                     await assistant_instance._safe_websocket_send({
#                         'type': 'warning',
#                         'content': 'Audio connection timeout. Will reconnect when audio resumes.',
#                         'error_code': 'DEEPGRAM_TIMEOUT_RECONNECTING'
#                     })
#                 except:
#                     pass
#                 return
            
#             # Handle other errors
#             if "insufficient credits" in error_message or "quota exceeded" in error_message:
#                 try:
#                     await assistant_instance._safe_websocket_send({
#                         'type': 'error',
#                         'content': 'Deepgram API credits exhausted.',
#                         'error_code': 'DEEPGRAM_CREDITS_EXHAUSTED'
#                     })
#                 except:
#                     pass
#             elif "unauthorized" in error_message or "invalid api key" in error_message:
#                 try:
#                     await assistant_instance._safe_websocket_send({
#                         'type': 'error',
#                         'content': 'Deepgram API key is invalid.',
#                         'error_code': 'DEEPGRAM_AUTH_ERROR'
#                     })
#                 except:
#                     pass

#         async def on_open(self_handler, open, **kwargs):
#             print(f"Deepgram connection opened successfully")

#         async def on_unhandled(self_handler, unhandled, **kwargs):
#             print(f"Unhandled Deepgram message: {unhandled}")

#         # Create Deepgram connection
#         if self.deepgram_connection_lost or self.pending_reconnection:
#             # Use reconnection method if this is a reconnection
#             print("Using reconnection method...")
#             dg_connection = await self._reconnect_deepgram()
#             if dg_connection is None:
#                 raise Exception("Reconnection failed")
#             # Clear pending reconnection flag
#             self.pending_reconnection = False
            
#             # Send any pending audio data that was buffered during disconnection
#             if hasattr(self, '_pending_audio_data') and self._pending_audio_data:
#                 print(f"Sending {len(self._pending_audio_data)} buffered audio chunks to Deepgram")
#                 for audio_chunk in self._pending_audio_data:
#                     try:
#                         await dg_connection.send(audio_chunk)
#                     except Exception as send_error:
#                         print(f"Error sending buffered audio: {send_error}")
#                         # Clear the buffer even if send fails to avoid infinite retry
#                         break
#                 # Clear the buffer after sending
#                 self._pending_audio_data = []
#         else:
#             # Initial connection
#             try:
#                 dg_connection = self.deepgram.listen.asyncwebsocket.v('1')
#             except AttributeError:
#                 dg_connection = self.deepgram.listen.asynclive.v('1')
            
#             # Register event handlers
#             dg_connection.on(LiveTranscriptionEvents.Open, on_open)
#             dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
#             dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
#             dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)
#             dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
#             dg_connection.on(LiveTranscriptionEvents.Close, on_close)
#             dg_connection.on(LiveTranscriptionEvents.Error, on_error)
#             dg_connection.on(LiveTranscriptionEvents.Unhandled, on_unhandled)

#             # Start the connection
#             try:
#                 connection_result = await dg_connection.start(self.dg_connection_options)
                
#                 if connection_result is False:
#                     raise Exception('Failed to connect to Deepgram with nova-2')
                
#                 # Reset connection flag and start keep-alive
#                 self.deepgram_connection_lost = False
#                 self.deepgram_keepalive_task = asyncio.create_task(self._deepgram_keepalive(dg_connection))
#                 self.last_audio_sent_time = datetime.now()
                
#             except Exception as connection_error:
#                 error_message = str(connection_error).lower()
#                 if "unauthorized" in error_message or "401" in error_message:
#                     await self.websocket.send_json({
#                         'type': 'error',
#                         'content': 'Deepgram API key is invalid or unauthorized.',
#                         'error_code': 'DEEPGRAM_UNAUTHORIZED'
#                     })
#                 elif "insufficient" in error_message or "credits" in error_message or "quota" in error_message:
#                     await self.websocket.send_json({
#                         'type': 'error',
#                         'content': 'Deepgram account has insufficient credits.',
#                         'error_code': 'DEEPGRAM_INSUFFICIENT_CREDITS'
#                     })
#                 else:
#                     await self.websocket.send_json({
#                         'type': 'error',
#                         'content': f'Deepgram connection error: {connection_error}',
#                         'error_code': 'DEEPGRAM_CONNECTION_ERROR'
#                     })
#                 raise Exception(f'Failed to connect to Deepgram: {connection_error}')

#         try:
#             # Track audio inactivity
#             last_audio_check = datetime.now()
#             audio_timeout = 3.0  # Reset audio_being_sent flag after 3 seconds of no audio
            
#             # Main message processing loop
#             while not self.finish_event.is_set() and not self.deepgram_connection_lost:
#                 # Check WebSocket state
#                 if self.websocket.client_state == WebSocketState.DISCONNECTED:
#                     self.finish_event.set()
#                     break
                
#                 # Check for audio inactivity
#                 current_time = datetime.now()
#                 if (self.audio_being_sent and self.last_audio_sent_time and 
#                     (current_time - self.last_audio_sent_time).total_seconds() > audio_timeout):
#                     print("Audio inactivity detected - resetting audio flag")
#                     self.audio_being_sent = False
                    
#                 # Receive message from client (with timeout to check audio inactivity)
#                 try:
#                     message = await asyncio.wait_for(self.websocket.receive(), timeout=1.0)
#                 except asyncio.TimeoutError:
#                     # Timeout is normal - continue to check audio inactivity
#                     continue
                    
#                 self.update_activity()
                
#                 # Handle different message types
#                 if message["type"] == "websocket.receive":
#                     if "bytes" in message:
#                         # Audio data - send to Deepgram
#                         audio_data = message["bytes"]
                        
#                         # Track audio activity
#                         if len(audio_data) > 0:
#                             self.audio_being_sent = True
#                             self.last_audio_sent_time = datetime.now()
                        
#                         try:
#                             await dg_connection.send(audio_data)
#                         except Exception as dg_send_error:
#                             error_msg = str(dg_send_error).lower()
#                             if "1011" in error_msg or "timeout" in error_msg or "connection" in error_msg:
#                                 print(f"Deepgram send error: {dg_send_error}")
#                                 self.deepgram_connection_lost = True
#                                 # Store this audio data for when we reconnect
#                                 if not hasattr(self, '_pending_audio_data'):
#                                     self._pending_audio_data = []
#                                 self._pending_audio_data.append(audio_data)
#                                 # Limit buffer size to prevent memory issues
#                                 if len(self._pending_audio_data) > 50:  # Keep last 50 chunks
#                                     self._pending_audio_data = self._pending_audio_data[-50:]
#                                 # Exit message loop to trigger reconnection
#                                 break
#                             else:
#                                 # For other errors, send message to client
#                                 try:
#                                     await self.websocket.send_json({
#                                         'type': 'error',
#                                         'content': 'Audio send error. Please try again.',
#                                         'error_code': 'DEEPGRAM_SEND_ERROR'
#                                     })
#                                 except:
#                                     pass
#                     elif "text" in message:
#                         # Handle frontend commands
#                         try:
#                             command = json.loads(message["text"])
#                             await self._handle_frontend_command(command)
#                         except:
#                             pass
                
#                 elif message["type"] == "websocket.disconnect":
#                     self.finish_event.set()
#                     break
                    
#         except WebSocketDisconnect:
#             self.finish_event.set()
#         except ConnectionClosedError:
#             self.finish_event.set()
#         except Exception as e:
#             error_msg = str(e).lower()
#             if "disconnect" in error_msg or "closed" in error_msg:
#                 self.finish_event.set()
#             elif "1011" in error_msg or "timeout" in error_msg:
#                 self.deepgram_connection_lost = True
#             else:
#                 print(f"Unexpected error in transcription session: {e}")
                
#         finally:
#             # Clean up this session
#             if hasattr(self, 'deepgram_keepalive_task') and self.deepgram_keepalive_task:
#                 self.deepgram_keepalive_task.cancel()
#                 try:
#                     await self.deepgram_keepalive_task
#                 except asyncio.CancelledError:
#                     pass
            
#             try:
#                 await dg_connection.finish()
#             except:
#                 pass

#             # Reset audio flag when session ends
#             if not self.pending_reconnection:
#                 self.audio_being_sent = False
                
#             # Clean up pending audio buffer
#             if hasattr(self, '_pending_audio_data'):
#                 self._pending_audio_data = []

#             # Session cleanup - only do this if we're really finishing
#             if self.finish_event.is_set():
#                 self.Object_session = {
#                     "Transcript": "",
#                     "Translation": "",
#                     "Original_Language": self.source_language,
#                     "Translated_Language": self.target_language,
#                     "Keywords": "",
#                     "Summary": ""
#                 }

#                 transcript_content = self.Get_Transcript()                
#                 self.Object_session["Transcript"] = transcript_content
#                 self.Complete_Transcript = transcript_content
#                 if self.Complete_Transcript:
#                     self.Object_session["Translation"] = await self.CompleteTranslation()

#                 if self.AI_mode == True:
#                     await self.Premium_Feature()

#                 # Finalize credit tracking and session
#                 await self.finalize_session_with_credits()

#                 # Save session data to database if user is identified
#                 await self.save_session_to_database()

#     async def manage_conversation(self):
        
#         while not self.finish_event.is_set():
#             try:
#                 # Check WebSocket connection status first
#                 if self.websocket.client_state == WebSocketState.DISCONNECTED:
#                     self.finish_event.set()
#                     break
                
#                 # Update activity periodically
#                 self.update_activity()
                
#                 # Small sleep to prevent busy waiting
#                 await asyncio.sleep(1)
                
#             except asyncio.CancelledError:
#                 break
#             except Exception as e:
#                 # Check if the error is related to connection issues
#                 error_msg = str(e).lower()
#                 if "disconnect" in error_msg or "closed" in error_msg:
#                     self.finish_event.set()
#                     break
#                 # Wait a bit before retrying
#                 await asyncio.sleep(1)

#     async def CompleteTranslation(self):
#         """Get complete translation using the selected translation provider and store it in self.Complete_Translation"""
#         #logger.info("Starting complete translation")
#         try:
#             transcript = self.Get_Transcript()
            
#             # Use the translation AI provider if available
#             # print("==========================>",self.translation_ai_provider.get_provider_name())
#             if self.translation_ai_provider:
#                 # For complete translation, we use non-streaming to get the full text at once
#                 translation = await self.translation_ai_provider.translate(
#                     transcript, self.target_language, self.system_prompt
#                 )
#                 if translation:
#                     self.Complete_Translation = translation
#                     return translation
#                 else:
#                     return None
            
#             elif self.translation_provider == 'claude' and self.claude_api_key:
#                 # Direct Claude API fallback when provider is not available
#                 #logger.warning("Claude translation provider not available, falling back to direct API")
#                 headers = {
#                     "x-api-key": self.claude_api_key,
#                     "Content-Type": "application/json",
#                     "anthropic-version": "2023-06-01"
#                 }
                
#                 # Use safe Claude model - if selected model doesn't start with 'claude', use default
#                 safe_model = self.selected_model if self.selected_model and self.selected_model.startswith('claude') else 'claude-3-5-sonnet-20241022'
                
#                 payload = {
#                     "model": safe_model,
#                     "max_tokens": 4000,
#                     "messages": [
#                         {"role": "user", "content": f"{self.system_prompt}\n\n{transcript}"}
#                     ],
#                     "temperature": 0.3
#                 }
                
#                 try:
#                     response = await self.http_client.post("https://api.anthropic.com/v1/messages", 
#                                                          json=payload, headers=headers, timeout=60.0)
#                     if response.status_code == 200:
#                         response_data = response.json()
#                         translation = response_data['content'][0]['text']
#                         self.Complete_Translation = translation
#                         #logger.info(f"Complete translation finished with Claude API. Length: {len(translation)}")
#                         return translation
#                     else:
#                         #logger.error(f"Claude API error during complete translation: {response.status_code}")
#                         return None
#                 except Exception as e:
#                     #logger.error(f"Error with Claude API fallback: {e}")
#                     return None       
#             elif self.translation_provider == 'gemini' and self.gemini_api_key:
#                 # Direct Gemini API fallback when provider is not available
#                 #logger.warning("Gemini translation provider not available, falling back to direct API")
                
#                 # Use safe Gemini model - if selected model doesn't start with 'gemini', use default
#                 safe_model = self.selected_model if self.selected_model and self.selected_model.startswith('gemini') else 'gemini-1.5-pro-latest'
                
#                 headers = {
#                     "Content-Type": "application/json"
#                 }
                
#                 payload = {
#                     "contents": [
#                         {
#                             "parts": [
#                                 {"text": f"{self.system_prompt}\n\n{transcript}"}
#                             ]
#                         }
#                     ],
#                     "generationConfig": {
#                         "temperature": 0.3,
#                         "maxOutputTokens": 4000
#                     }
#                 }
                
#                 try:
#                     url = f"https://generativelanguage.googleapis.com/v1beta/models/{safe_model}:generateContent?key={self.gemini_api_key}"
#                     response = await self.http_client.post(url, json=payload, headers=headers, timeout=60.0)
#                     if response.status_code == 200:
#                         response_data = response.json()
#                         translation = response_data['candidates'][0]['content']['parts'][0]['text']
#                         self.Complete_Translation = translation
#                         return translation
#                     else:
#                         return None
#                 except Exception as e:
#                     #logger.error(f"Error with Gemini API fallback: {e}")
#                     return None
#             elif self.translation_provider == 'openai' and self.openai_api_key:
#                 # Only fallback to OpenAI API when OpenAI is the selected provider
#                 #logger.warning("OpenAI translation provider not available, falling back to direct API")
#                 headers = {
#                     "Authorization": f"Bearer {self.openai_api_key}",
#                     "Content-Type": "application/json"
#                 }
                
#                 # Use safe OpenAI model - if selected model doesn't start with 'gpt', use default
#                 safe_model = self.selected_model if self.selected_model and self.selected_model.startswith('gpt') else CHAT_GPT_MODEL_BEING_USED
                
#                 payload = {
#                     "model": safe_model,
#                     "messages": [
#                         {"role": "system", "content": self.system_prompt},
#                         {"role": "user", "content": transcript}
#                     ],
#                     "temperature": 0.3,
#                     "stream": False
#                 }
                
#                 # Set a longer timeout for complete translation (60 seconds)
#                 #logger.info(f"Sending translation request for transcript of length: {len(self.Complete_Transcript)}")
#                 response = await self.http_client.post("https://api.openai.com/v1/chat/completions", 
#                                                      json=payload, headers=headers, timeout=60.0)
#                 if response.status_code == 200:
#                     response_data = response.json()
#                     translation = response_data['choices'][0]['message']['content']
#                     self.Complete_Translation = translation
#                     #logger.info(f"Complete translation finished. Length: {len(translation)}")
#                     return translation
#                 else:
#                     #logger.error(f"OpenAI API error during complete translation: {response.status_code}")
#                     response_text = response.text
#                     #logger.error(f"Response text: {response_text}")
#                     await self.handle_openai_error(response.status_code, response_text)
#                     return None
#             else:
#                 #logger.error(f"No {self.translation_provider} translation provider available")
#                 return None
                    
#         except httpx.TimeoutException as e:
#             #logger.error(f"Timeout during complete translation: {e}")
#             return None
#         except Exception as e:
#             #logger.error(f"Error during complete translation: {e}")
#             return None

#     def Get_Transcript(self):
#         """Get a time-ordered transcript as a formatted string."""
#         if not self.speaker_segments:
#             # If no speaker segments available, return the complete transcript as is
#             return self.Complete_Transcript
        
#         # Sort speaker segments by start time to get chronological order
#         sorted_segments = sorted(self.speaker_segments, key=lambda x: x['start_time'])
        
#         # Build formatted transcript with time-ordered speaker labels
#         formatted_transcript = ""
#         for segment in sorted_segments:
#             if segment['transcript'].strip():
#                 formatted_transcript += f"Speaker {segment['speaker_id']}: {segment['transcript'].strip()}\n"
        
#         # If we have speaker data, also update Complete_Transcript for consistency
#         if formatted_transcript:
#             self.Complete_Transcript = formatted_transcript
            
#         return formatted_transcript or self.Complete_Transcript

#     async def save_session_to_database(self):
#         """Save the current session data to the database for the identified user."""
#         try:
#             if not self.user_email:
#                 return False
            
#             # Prevent duplicate saves
#             if hasattr(self, 'session_saved') and self.session_saved:
#                 return True
            
#             # Check if session has meaningful content before saving
#             transcript_content = self.Object_session.get("Transcript", "").strip()
#             translation_content = self.Object_session.get("Translation", "").strip()
            
#             # Don't save empty sessions - require at least some transcript content
#             if not transcript_content or len(transcript_content) < 10:
#                 return False
            
#             # Get current date for session storage
#             # Prepare session data in the expected format
#             session_data = {
#                 "Original Text": transcript_content,
#                 "Translated Text": translation_content,
#                 "Summary": self.Object_session.get("Summary", ""),
#                 "Original Language": self.languages[0],
#                 "Translated Language": self.languages[1],
#                 "Keywords": self.Object_session.get("Keywords", "")
#             }
            
#             # Find the user in the database
#             user = Find_User_DB(self.user_email)
#             if not user:
#                 return False
            
#             # Initialize session structure if needed
#             current_date = self.date
#             current_time = self.time
            
#             if "Session" not in user:
#                 user["Session"] = {}
#             if current_date not in user["Session"]:
#                 user["Session"][current_date] = {}
#             if self.web_page_name not in user["Session"][current_date]:
#                 user["Session"][current_date][self.web_page_name] = {}
            
#             # Save the session data with timestamp
#             user["Session"][current_date][self.web_page_name][current_time] = session_data

            
#             # Update the database
#             result = collection.update_one(
#                 {"Email": self.user_email}, 
#                 {"$set": {"Session": user["Session"]}}
#             )
            
#             if result.modified_count > 0:
#                 # Mark session as saved to prevent duplicates
#                 self.session_saved = True
#                 return True
#             else:
#                 return False
                
#         except Exception as e:
#             return False

#     async def cleanup(self):
#         """Clean up any resources used by the Assistant."""
#         #logger.info("Cleaning up Assistant resources")
        
#         # Set the finish event to signal all tasks to stop
#         self.finish_event.set()
        
#         # Finalize session with credits before cleanup (if not already done)
#         try:
#             if not self.session_finalized:
#                 await self.finalize_session_with_credits()
#         except Exception as e:
#             pass
        
#         # Clean up AI providers
#         if hasattr(self, 'translation_ai_provider') and self.translation_ai_provider:
#             try:
#                 await self.translation_ai_provider.cleanup()
#             except Exception as e:
#                 pass
                
#         if hasattr(self, 'ai_features_provider') and self.ai_features_provider:
#             try:
#                 await self.ai_features_provider.cleanup()
#             except Exception as e:
#                 pass
        
#         # Clean up OpenAI WebSocket if it exists
#         if hasattr(self, 'openai_ws') and self.openai_ws:
#             try:
#                 await self.openai_ws.close()
#             except Exception as e:
#                 pass
        
#         # Clean up WebSocket
#         if self.websocket:
#             try:
#                 if self.websocket.client_state != WebSocketState.DISCONNECTED:
#                     await safe_websocket_close(self.websocket)
#             except Exception as e:
#                 pass
#         # Clean up HTTP client
#         if self.http_client:
#             try:
#                 await self.http_client.aclose()
#             except Exception as e:
#                 pass

#     async def keep_alive_ping(self):
#         """Send periodic keep-alive messages to maintain WebSocket connection"""
#         try:
#             while not self.finish_event.is_set():
#                 try:
#                     # Check if connection is still alive
#                     if self.websocket.client_state == WebSocketState.DISCONNECTED:
#                         break
                    
#                     # Check for inactivity timeout
#                     time_since_activity = (datetime.now() - self.last_activity).total_seconds()
#                     if time_since_activity > self.connection_timeout:
#                         #logger.warning(f"Connection timeout after {time_since_activity} seconds of inactivity")
#                         await self.websocket.send_json({
#                             'type': 'timeout_warning',
#                             'message': 'Connection will be closed due to inactivity'
#                         })
#                         await asyncio.sleep(10)  # Give 10 seconds grace period
#                         break
                    
#                     # Send keep-alive ping
#                     await self.websocket.send_json({
#                         'type': 'keep_alive',
#                         'timestamp': datetime.now().isoformat()
#                     })
#                     #logger.debug("Keep-alive ping sent")
                    
#                 except WebSocketDisconnect:
#                     break
#                 except Exception as e:
#                     break
                
#                 # Wait for next ping interval
#                 await asyncio.sleep(self.keep_alive_interval)
                
#         except asyncio.CancelledError:
#             pass
#         except Exception as e:
#             pass

#     def update_activity(self):
#         """Update last activity timestamp"""
#         self.last_activity = datetime.now()
    
#     def _handle_translation_task_completion(self, task):
#         """Handle completion of translation tasks and log any exceptions."""
#         if task.exception():
#             pass
#         elif task.cancelled():
#             pass
#         else:
#             pass
    
#     async def _deepgram_keepalive(self, dg_connection):
#         """Send periodic keep-alive messages to maintain Deepgram connection."""
#         try:
#             while not self.finish_event.is_set() and not self.deepgram_connection_lost:
#                 await asyncio.sleep(1)  # Send keep-alive every 1 seconds
#                 if not self.deepgram_connection_lost and not self.finish_event.is_set():
#                     try:
#                         # Send a small keep-alive message (empty JSON message)
#                         await dg_connection.send("{\"type\":\"KeepAlive\"}")
#                         print("Sent Deepgram keep-alive")
#                     except Exception as e:
#                         print(f"Keep-alive failed: {e}")
#                         # If keep-alive fails, mark connection as lost
#                         self.deepgram_connection_lost = True
#                         break
#         except asyncio.CancelledError:
#             print("Keep-alive task cancelled")
#             pass
#         except Exception as e:
#             print(f"Keep-alive error: {e}")
#             pass
    
#     async def initialize_credit_tracking(self):
#         """Initialize credit tracking for paid users only"""
#         try:
#             if not self.user_email:
#                 return True  # Allow session for free users without email
            
#             # Check if user is a paid user
#             from User import User
#             user = User()
#             user_data = user.Get_User_Data(self.user_email)
#             is_paid_user = user_data.get('Paid_User', False) if user_data else False
            
#             if not is_paid_user:
#                 return True  # Allow session for free users
            
            
#             # Get current credits and initialize tracking
#             current_credits = credit_manager.get_user_credits(self.user_email)
#             model = self.selected_model or 'gpt-4o-mini'  # Default model
            
#             # Initialize tracking without checking credits upfront
#             self.session_start_time = datetime.now()
#             self.last_credit_check_time = self.session_start_time
            
#             # Send initial credit status
#             await self.websocket.send_json({
#                 'type': 'credit_status',
#                 'current_balance': current_credits,
#                 'model': model,
#                 'cost_per_minute': credit_manager.calculate_total_cost_per_minute(model)
#             })
            
#             return True
            
#         except Exception as e:
#             #logger.error(f"Error initializing credit tracking: {e}")
#             return False
    
#     async def start_credit_monitoring(self):
#         """Start the credit monitoring task for paid users only"""
#         try:
#             if not self.user_email or not self.selected_model:
#                 return
            
#             # Check if user is a paid user
#             from User import User
#             user = User()
#             user_data = user.Get_User_Data(self.user_email)
#             is_paid_user = user_data.get('Paid_User', False) if user_data else False
            
#             if is_paid_user:
#                 self.credit_task = asyncio.create_task(self.monitor_credits())
#             else:
#                 pass
#         except Exception as e:
#             pass
    
#     async def monitor_credits(self):
#         """Monitor and deduct credits every minute"""
#         try:
#             while not self.finish_event.is_set():
#                 await asyncio.sleep(self.credit_check_interval)
                
#                 if self.finish_event.is_set():
#                     break
                
#                 await self.process_credit_deduction()
                
#         except asyncio.CancelledError:
#             pass
#         except Exception as e:
#             pass
    
#     async def process_credit_deduction(self):
#         """Process per-minute credit deduction"""
#         try:
#             if not self.user_email or not self.selected_model:
#                 return
            
#             current_time = datetime.now()
            
#             # Calculate minutes elapsed since last check
#             if self.last_credit_check_time:
#                 time_diff = (current_time - self.last_credit_check_time).total_seconds() / 60.0
#             else:
#                 time_diff = 1.0  # Default to 1 minute
            
#             # Calculate credits to deduct using the new total cost calculation
#             cost_per_minute = credit_manager.calculate_total_cost_per_minute(self.selected_model)
#             credits_to_deduct = cost_per_minute * time_diff
            
#             # Check if user has sufficient credits
#             sufficient, current_balance, required = credit_manager.check_sufficient_credits(
#                 self.user_email, self.selected_model, time_diff
#             )
            
#             if not sufficient:
#                 # Insufficient credits - end session
#                 await self.handle_insufficient_credits(current_balance, required)
#                 return
            
#             # Deduct credits with model info and disable auto-logging (we'll log once per session)
#             success, new_balance, message = credit_manager.deduct_credits(
#                 self.user_email, credits_to_deduct, f"transcription_{self.selected_model}", 
#                 model=self.selected_model, auto_log=False
#             )
            
#             if success:
#                 self.credits_deducted += credits_to_deduct
#                 self.total_session_duration += time_diff
#                 self.last_credit_check_time = current_time
                
#                 # Send credit update to frontend
#                 await self.websocket.send_json({
#                     'type': 'credit_update',
#                     'credits_deducted': credits_to_deduct,
#                     'new_balance': new_balance,
#                     'session_duration': self.total_session_duration,
#                     'total_session_cost': self.credits_deducted
#                 })
                
            
#         except Exception as e:
#             pass
    
#     async def finalize_session_with_credits(self):
#         """Finalize session with credit information for paid users only"""
#         try:
#             # Prevent duplicate finalization
#             if self.session_finalized:
#                 return
#             self.session_finalized = True
            
#             if not self.user_email:
#                 return
            
#             # Check if user is a paid user
#             from User import User
#             user = User()
#             user_data = user.Get_User_Data(self.user_email)
#             is_paid_user = user_data.get('Paid_User', False) if user_data else False
            
#             if not is_paid_user:
#                 # For free users, just update session data without credit tracking
#                 if hasattr(self, 'Object_session'):
#                     self.Object_session['Credits_Used'] = 0.0
#                     self.Object_session['Session_Duration_Minutes'] = 0.0
#                     self.Object_session['Model_Used'] = self.selected_model or 'free_tier'
#                 return
            
#             # Calculate final session cost
#             if self.session_start_time:
#                 session_end_time = datetime.now()
#                 total_duration = (session_end_time - self.session_start_time).total_seconds() / 60.0
                
#                 # Ensure we deduct for any remaining time (avoid double deduction)
#                 if self.last_credit_check_time and not self.finish_event.is_set():
#                     remaining_time = (session_end_time - self.last_credit_check_time).total_seconds() / 60.0
#                     # Only deduct if more than 6 seconds remaining and not already processed
#                     if remaining_time > 0.1 and remaining_time < 10.0:  # Cap at 10 minutes to prevent errors
#                         await self.process_final_credit_deduction(remaining_time)
            
#             # Process AI mode credits BEFORE updating session data and logging
#             ai_feature_cost = 0.0
#             if self.AI_mode == True:
#                 try:
#                     ai_feature_cost = credit_manager.Calculate_Cost_of_Premium_Feature(
#                         self.selected_model, len(self.Complete_Transcript), 
#                         len(self.Object_session.get("Summary", "")), 
#                         len(str(self.Object_session.get("Keywords", "")))
#                     )
                    
#                     # Check if user has sufficient credits for AI features
#                     current_balance = credit_manager.get_user_credits(self.user_email)
#                     if current_balance >= ai_feature_cost:
#                         # Deduct AI feature credits
#                         success, new_balance, message = credit_manager.deduct_credits(
#                             self.user_email, ai_feature_cost, f"ai_features_{self.selected_model}",
#                             model=self.selected_model, auto_log=False
#                         )
#                         if success:
#                             self.credits_deducted += ai_feature_cost
#                         else:
#                             ai_feature_cost = 0.0  # Reset if deduction failed
#                     else:
#                         ai_feature_cost = 0.0  # Not enough credits for AI features
                        
#                 except Exception as e:
#                     ai_feature_cost = 0.0
            
#             # Update session data with complete credit information
#             if hasattr(self, 'Object_session'):
#                 self.Object_session['Credits_Used'] = self.credits_deducted
#                 self.Object_session['Session_Duration_Minutes'] = self.total_session_duration
#                 self.Object_session['Model_Used'] = self.selected_model
#                 if ai_feature_cost > 0:
#                     self.Object_session['Premium_Feature_Cost'] = ai_feature_cost
            
#             # Generate session ID for tracking
#             session_id = f"{self.user_email}_{self.time}_{self.date}"

#             # Log session-level credit transaction with complete credit information
#             if self.credits_deducted > 0:
#                 credit_manager.log_session_credits(
#                     email=self.user_email,
#                     total_credits_used=self.credits_deducted,
#                     model=self.selected_model,
#                     session_duration=self.total_session_duration,
#                     platform=self.web_page_name or 'Unknown',
#                     session_id=session_id,
#                     time=self.time,
#                     date=self.date
#                 )
            
#         except Exception as e:
#             pass
    
#     async def process_final_credit_deduction(self, remaining_time):
#         """Process final credit deduction for remaining session time"""
#         try:
#             if remaining_time <= 0:
#                 return
            
#             cost_per_minute = credit_manager.calculate_total_cost_per_minute(self.selected_model)
#             final_credits = cost_per_minute * remaining_time
            
#             # Check and deduct final credits
#             sufficient, current_balance, required = credit_manager.check_sufficient_credits(
#                 self.user_email, self.selected_model, remaining_time
#             )
            
#             if sufficient:
#                 # Deduct final credits with model info and disable auto-logging
#                 success, new_balance, message = credit_manager.deduct_credits(
#                     self.user_email, final_credits, f"final_transcription_{self.selected_model}",
#                     model=self.selected_model, auto_log=False
#                 )
                
#                 if success:
#                     self.credits_deducted += final_credits
#                     self.total_session_duration += remaining_time
                    
#                     # Send final credit update
#                     await self.websocket.send_json({
#                         'type': 'session_finalized',
#                         'final_credits_deducted': final_credits,
#                         'total_credits_used': self.credits_deducted,
#                         'final_balance': new_balance,
#                         'total_duration_minutes': self.total_session_duration
#                     })
                    
            
#         except Exception as e:
#             pass
    
#     async def handle_insufficient_credits(self, current_balance, required_credits):
#         """Handle when user runs out of credits during transcription"""
#         try:
#             # Stop transcription
#             self.finish_event.set()
            
#             # Send insufficient credits message
#             await self.websocket.send_json({
#                 'type': 'credit_insufficient',
#                 'content': f'Credits exhausted during transcription. Session ended.',
#                 'error_code': 'CREDITS_EXHAUSTED',
#                 'current_balance': current_balance,
#                 'required_credits': required_credits,
#                 'session_duration': self.total_session_duration,
#                 'total_credits_used': self.credits_deducted
#             })
            
#             # Save final session data
#             await self.finalize_session_with_credits()
            
            
#         except Exception as e:
#             pass

#     async def run(self):
#         try:
#             # Initialize credit tracking
#             if self.user_email:
#                 await self.initialize_credit_tracking()
            
#             self.keep_alive_task = asyncio.create_task(self.keep_alive_ping())
           
#             await self.start_credit_monitoring()
            
#             if self.translation_provider == 'openai' and self.selected_model == 'gpt-4o-realtime-preview':
#                 # Only use WebSocket for OpenAI realtime model which requires separate response processing
#                 await self.translation_ai_provider.connect_to()
                
#                 async with asyncio.TaskGroup() as tg:
#                     tg.create_task(self.transcribe_audio())
#                     tg.create_task(self.manage_conversation())
#                     tg.create_task(self.process_openai_responses())
#             elif self.translation_provider == 'gemini':
                
#                 # Try to connect to Gemini Live API
#                 try:
#                     await self.translation_ai_provider.connect_to()
#                     self.gemini_ws = self.translation_ai_provider.gemini_ws
                    
#                     # Test the WebSocket connection with a simple message
#                     try:
#                         print("Testing Gemini WebSocket connection...")
#                         await self.translation_ai_provider.send_translation_request("Hello", "You are a translator. Translate this text to English.")
#                         print("Test message sent successfully")
                        
#                         # If successful, use WebSocket approach
#                         async with asyncio.TaskGroup() as tg:
#                             tg.create_task(self.transcribe_audio())
#                             tg.create_task(self.manage_conversation())
#                             tg.create_task(self.process_gemini_responses())
                            
#                     except Exception as e:
#                         print(f"Gemini WebSocket test failed: {e}")
#                         print("Falling back to REST API approach")
#                         # Fall back to REST API approach
#                         async with asyncio.TaskGroup() as tg:
#                             tg.create_task(self.transcribe_audio())
#                             tg.create_task(self.manage_conversation())
                            
#                 except Exception as e:
#                     print(f"Failed to connect to Gemini Live API: {e}")
#                     print("Using REST API approach for Gemini")
#                     # Use REST API approach
#                     async with asyncio.TaskGroup() as tg:
#                         tg.create_task(self.transcribe_audio())
#                         tg.create_task(self.manage_conversation())
#             else:
#                 # For all other providers (including OpenAI REST API, Claude, and Gemini)
#                 async with asyncio.TaskGroup() as tg:
#                     tg.create_task(self.transcribe_audio())
#                     tg.create_task(self.manage_conversation())
                    
        
#         except asyncio.CancelledError:
#             pass
#         except Exception as e:
#             pass
#         finally:
                
#             # Clean up credit monitoring task
#             if hasattr(self, 'credit_task') and self.credit_task and not self.credit_task.done():
#                 self.credit_task.cancel()
#                 try:
#                     await self.credit_task
#                 except asyncio.CancelledError:
#                     pass
#             # Clean up keep-alive task
#             if hasattr(self, 'keep_alive_task') and self.keep_alive_task and not self.keep_alive_task.done():
#                 self.keep_alive_task.cancel()
#                 try:
#                     await self.keep_alive_task
#                 except asyncio.CancelledError:
#                     pass
            
#             # Call comprehensive cleanup
#             await self.cleanup()
          

import asyncio
import json
import httpx
import re
import string
from starlette.websockets import WebSocketDisconnect, WebSocketState
from deepgram import (
    DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents, LiveOptions)
import websockets
from languages import language_codes
from datetime import datetime
from Autentication import Find_User_DB, collection
from payment import credit_manager



CHAT_GPT_MODEL_BEING_USED='gpt-4o-realtime-preview'

deepgram_config = DeepgramClientOptions(options={'keepalive': 'true'})

class Assistant:
    def __init__(self, websocket, dg_api_key, openai_api_key, source_language, target_language, mode='speed', translation_provider='openai', claude_api_key=None, gemini_api_key=None, user_email=None, ai_mode=False, ai_provider='openai', web_page_name='Unknown', selected_model=None, Time=None, Date=None):    
        self.languages = [source_language,target_language]
        self.websocket = websocket
        self.transcript_parts = []
        self.transcript_queue = asyncio.Queue()
        self.finish_event = asyncio.Event()
        self.openai_ws = None
        self.gemini_ws = None
        self.source_language = language_codes[source_language]
        self.target_language = target_language
        self.openai_api_key = openai_api_key
        self.claude_api_key = claude_api_key
        self.gemini_api_key = gemini_api_key
        self.dg_api_key = dg_api_key
        self.translation_provider = translation_provider.lower()
        self.user_email = user_email
        self.AI_mode = ai_mode
        self.ai_provider = ai_provider.lower()  
        self.web_page_name = web_page_name
        self.selected_model = selected_model

        # Time and Date
        self.time = Time
        self.date = Date
        
        # Credit tracking attributes
        self.session_start_time = None
        self.last_credit_check_time = None
        self.credit_check_interval = 60.0  # Check credits every minute
        self.credit_task = None
        self.total_session_duration = 0.0  # Track total session duration in minutes
        self.credits_deducted = 0.0  # Track total credits deducted this session
        self.session_finalized = False  # Flag to prevent duplicate session finalization
        
        # Keep-alive mechanism attributes
        self.keep_alive_interval = 30  # Send keep-alive every 30 seconds
        self.keep_alive_task = None
        self.last_activity = datetime.now()
        self.connection_timeout = 300  # 5 minutes timeout for inactivity
        
        # Translation status tracking
        self.translation_enabled = True  # Track if translation is working
        self.consecutive_translation_failures = 0  # Track consecutive failures
        self.max_translation_failures = 3  # Disable translation after 3 consecutive failures
        
        # Deepgram connection management
        self.deepgram_connection_lost = False
        self.last_audio_sent_time = None
        self.deepgram_keepalive_task = None
        self.audio_being_sent = False  # Track if audio is actively being sent
        self.pending_reconnection = False  # Track if reconnection is needed when audio resumes
        self.last_transcript_timestamp = None  # Track the timestamp of the last received transcript
        
        self.dg_connection_options = LiveOptions(
                            model="nova-2",
                            language=self.source_language,
                            smart_format=True,
                            interim_results=True,
                            utterance_end_ms="1000",
                            vad_events=True,
                            endpointing=300,
                            diarize=True,
                            punctuate=True,
                        )
        
        self.mode = mode

        # Session management flags
        self.session_saved = False
        self.session_initialized = False
        self.cleanup_completed = False
        # Initialize session data structure
        self.Object_session = {
            "Transcript": "",
            "Translation": "",
            "Summary": "",
            "Keywords": "",
            "Original Language": source_language,
            "Translated Language": target_language,
            "Model_Used": selected_model,
            "Credits_Used": 0.0,
            "Session_Duration_Minutes": 0.0,
            "Session_Start_Time": None,
            "Session_End_Time": None,
            "Web_Page": web_page_name,
            "AI_Mode": ai_mode,
            "AI_Provider": ai_provider
        }

        
        # Force gpt-4o-mini model when in speed mode for faster responses
        if mode == 'speed' and translation_provider == 'openai':
            self.selected_model = CHAT_GPT_MODEL_BEING_USED
            #logger.info(f"Speed mode detected: forcing model to {CHAT_GPT_MODEL_BEING_USED} for optimal performance")
        elif not selected_model:
            # Default fallback model if none selected
            self.selected_model = CHAT_GPT_MODEL_BEING_USED
            #logger.info(f"No model specified: defaulting to {CHAT_GPT_MODEL_BEING_USED}")
        else:
            self.selected_model = selected_model
            
        self.Object_session = {}
        self.Complete_Transcript = ""
        self.Complete_Translation = ""
        # Speaker diarization attributes
        self.speaker_segments = []  # List of speaker segments with timestamps and speaker IDs
        self.current_speaker_segment = None  # Current ongoing speaker segment
        self.speaker_transcript = {}  # Dictionary mapping speaker IDs to their complete transcripts
        self.system_prompt = f"""You are a helpful translator whose sole purpose is to generate {target_language} translation of provided text. Do not say anything else. You will return plain {target_language} translation of the provided text and nothing else. Translate the text provided below, which is in a diarized form, preserving the diarized format:"""
        self.system_prompt_summarizer = f"""
        You are a helpful summarizer whose sole purpose is to generate a summary of the provided text in {target_language}. Do not say anything else. You will not answer to any user question, you will just summarize it. No matter, whatever the user says, you will only summarize it and not respond to what user said.
        Generate an insightful summary of the provided text in {target_language}. Ensure that the summary strictly reflects the original content without adding, omitting, or altering any information or the meaning of the words.

        # Steps

        1. Carefully read the entire input text to fully understand its content.
        2. Identify the key points and essential information presented in the text.
        3. Summarize these key points clearly and concisely in {target_language}.
        4. Avoid introducing any new information or biased interpretations.
        5. Preserve the original intent and meaning of the text throughout the summary.

        # Output Format

        - Provide the summary as a coherent paragraph.
        - The summary must be in {target_language}.
        - Ensure the summary is faithful to the original text, containing no distortions or omissions.

        # Notes

        - Do not add personal opinions or external information.
        - Avoid paraphrasing that changes the meaning of the original text.
        - Maintain neutrality and clarity.
        - Always respond in {target_language}.


        """
       
        self.system_prompt_keywords = """
        You are a keyword extractor. Extract the most relevant keywords and phrases from the following text. For each keyword:
        1. Find single and multi-word keywords that capture important concepts
        2. Include the starting position (index) where each keyword appears in the text
        3. Assign a relevance score between 0 and 1 for each keyword
        4. Assign a sentiment score between -1 and 1 for each keyword
        5. Focus on nouns, noun phrases, and important terms

        Return the results as a JSON array in this exact format:
        {{
        "keywords": [
            {{
            "keyword": "example term",
            "positions": [5],
            "score": 0.95,
            "sentiment": 0.95
            }},
            {{
            "keyword": "another keyword",
            "positions": [20],
            "score": 0.85,
            "sentiment": -0.32
            }}
        ]
        }}

        Important:
        - Each keyword must have its EXACT character position in the text (counting from 0)
        - Scores should reflect the relevance (0–1)
        - Include both single words and meaningful phrases
        - List results from highest to lowest score
        - Sentiment should reflect the relevance (-1 to 1)
        """

        # Initialize Deepgram client only if API key is provided
        if dg_api_key and dg_api_key.strip() and dg_api_key != "your_deepgram_api_key_here":
            try:
                self.deepgram = DeepgramClient(dg_api_key, config=deepgram_config)
            except Exception as e:
                print(f"Error initializing Deepgram client: {e}")
                self.deepgram = None
        else:
            print("Warning: No valid Deepgram API key provided")
            self.deepgram = None     
        self.stime = 0
        
    # Credit Tracking
    def is_paid_user(self):
        """Check if the current user is a paid user"""
        if not self.user_email:
            return False
        try:
            from User import User
            user = User()
            user_data = user.Get_User_Data(self.user_email)
            return user_data.get('Paid_User', False) if user_data else False
        except Exception as e:
            return False
    
    @property
    def user_credits(self):
        """Get current user credits from credit manager - only for paid users"""
        if not self.user_email:
            return None
        
        # Only return credits for paid users
        if not self.is_paid_user():
            return None
            
        try:
            from payment import credit_manager
            return credit_manager.get_user_credits(self.user_email)
        except Exception as e:
            #logger.error(f"Error getting user credits: {e}")
            return None

    async def initialize_credit_tracking(self):
        """Initialize credit tracking for paid users only"""
        try:
            if not self.user_email:
                return True  # Allow session for free users without email
            
            # Check if user is a paid user
            from User import User
            user = User()
            user_data = user.Get_User_Data(self.user_email)
            is_paid_user = user_data.get('Paid_User', False) if user_data else False
            
            if not is_paid_user:
                return True  # Allow session for free users
            
            
            # Get current credits and initialize tracking
            current_credits = credit_manager.get_user_credits(self.user_email)
            model = self.selected_model or 'gpt-4o-mini'  # Default model
            
            # Initialize tracking without checking credits upfront
            self.session_start_time = datetime.now()
            self.last_credit_check_time = self.session_start_time
            
            # Send initial credit status
            await self.websocket.send_json({
                'type': 'credit_status',
                'current_balance': current_credits,
                'model': model,
                'cost_per_minute': credit_manager.calculate_total_cost_per_minute(model)
            })
            
            return True
            
        except Exception as e:
            #logger.error(f"Error initializing credit tracking: {e}")
            return False

    async def start_credit_monitoring(self):
        """Start the credit monitoring task for paid users only"""
        try:
            if not self.user_email or not self.selected_model:
                return
            
            # Check if user is a paid user
            from User import User
            user = User()
            user_data = user.Get_User_Data(self.user_email)
            is_paid_user = user_data.get('Paid_User', False) if user_data else False
            
            if is_paid_user:
                self.credit_task = asyncio.create_task(self.monitor_credits())
            else:
                pass
        except Exception as e:
            pass
    
    async def monitor_credits(self):
        """Monitor and deduct credits every minute"""
        try:
            while not self.finish_event.is_set():
                await asyncio.sleep(self.credit_check_interval)
                
                if self.finish_event.is_set():
                    break
                
                await self.process_credit_deduction()
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            pass
    
    async def process_credit_deduction(self):
        """Process per-minute credit deduction"""
        try:
            if not self.user_email or not self.selected_model:
                return
            
            current_time = datetime.now()
            
            # Calculate minutes elapsed since last check
            if self.last_credit_check_time:
                time_diff = (current_time - self.last_credit_check_time).total_seconds() / 60.0
            else:
                time_diff = 1.0  # Default to 1 minute
            
            # Calculate credits to deduct using the new total cost calculation
            cost_per_minute = credit_manager.calculate_total_cost_per_minute(self.selected_model)
            credits_to_deduct = cost_per_minute * time_diff
            
            # Check if user has sufficient credits
            sufficient, current_balance, required = credit_manager.check_sufficient_credits(
                self.user_email, self.selected_model, time_diff
            )
            
            if not sufficient:
                # Insufficient credits - end session
                await self.handle_insufficient_credits(current_balance, required)
                return
            
            # Deduct credits with model info and disable auto-logging (we'll log once per session)
            success, new_balance, message = credit_manager.deduct_credits(
                self.user_email, credits_to_deduct, f"transcription_{self.selected_model}", 
                model=self.selected_model, auto_log=False
            )
            
            if success:
                self.credits_deducted += credits_to_deduct
                self.total_session_duration += time_diff
                self.last_credit_check_time = current_time
                
                # Send credit update to frontend
                await self.websocket.send_json({
                    'type': 'credit_update',
                    'credits_deducted': credits_to_deduct,
                    'new_balance': new_balance,
                    'session_duration': self.total_session_duration,
                    'total_session_cost': self.credits_deducted
                })
                
            
        except Exception as e:
            pass
    
    async def finalize_session_with_credits(self):
        """Finalize session with credit information for paid users only"""
        try:
            # Prevent duplicate finalization
            if self.session_finalized:
                return
            self.session_finalized = True
            
            if not self.user_email:
                return
            
            # Check if user is a paid user
            from User import User
            user = User()
            user_data = user.Get_User_Data(self.user_email)
            is_paid_user = user_data.get('Paid_User', False) if user_data else False
            
            if not is_paid_user:
                # For free users, just update session data without credit tracking
                if hasattr(self, 'Object_session'):
                    self.Object_session['Credits_Used'] = 0.0
                    self.Object_session['Session_Duration_Minutes'] = 0.0
                    self.Object_session['Model_Used'] = self.selected_model or 'free_tier'
                return
            
            # Calculate final session cost
            if self.session_start_time:
                session_end_time = datetime.now()
                total_duration = (session_end_time - self.session_start_time).total_seconds() / 60.0
                
                # Ensure we deduct for any remaining time (avoid double deduction)
                if self.last_credit_check_time and not self.finish_event.is_set():
                    remaining_time = (session_end_time - self.last_credit_check_time).total_seconds() / 60.0
                    # Only deduct if more than 6 seconds remaining and not already processed
                    if remaining_time > 0.1 and remaining_time < 10.0:  # Cap at 10 minutes to prevent errors
                        await self.process_final_credit_deduction(remaining_time)
            
            # Process AI mode credits BEFORE updating session data and logging
            ai_feature_cost = 0.0
            if self.AI_mode == True:
                try:
                    ai_feature_cost = credit_manager.Calculate_Cost_of_Premium_Feature(
                        self.selected_model, len(self.Complete_Transcript), 
                        len(self.Object_session.get("Summary", "")), 
                        len(str(self.Object_session.get("Keywords", "")))
                    )
                    
                    # Check if user has sufficient credits for AI features
                    current_balance = credit_manager.get_user_credits(self.user_email)
                    if current_balance >= ai_feature_cost:
                        # Deduct AI feature credits
                        success, new_balance, message = credit_manager.deduct_credits(
                            self.user_email, ai_feature_cost, f"ai_features_{self.selected_model}",
                            model=self.selected_model, auto_log=False
                        )
                        if success:
                            self.credits_deducted += ai_feature_cost
                        else:
                            ai_feature_cost = 0.0  # Reset if deduction failed
                    else:
                        ai_feature_cost = 0.0  # Not enough credits for AI features
                        
                except Exception as e:
                    ai_feature_cost = 0.0
            
            # Update session data with complete credit information
            if hasattr(self, 'Object_session'):
                self.Object_session['Credits_Used'] = self.credits_deducted
                self.Object_session['Session_Duration_Minutes'] = self.total_session_duration
                self.Object_session['Model_Used'] = self.selected_model
                if ai_feature_cost > 0:
                    self.Object_session['Premium_Feature_Cost'] = ai_feature_cost
            
            # Generate session ID for tracking
            session_id = f"{self.user_email}_{self.time}_{self.date}"

            # Log session-level credit transaction with complete credit information
            if self.credits_deducted > 0:
                credit_manager.log_session_credits(
                    email=self.user_email,
                    total_credits_used=self.credits_deducted,
                    model=self.selected_model,
                    session_duration=self.total_session_duration,
                    platform=self.web_page_name or 'Unknown',
                    session_id=session_id,
                    time=self.time,
                    date=self.date
                )
            
        except Exception as e:
            pass
    
    async def process_final_credit_deduction(self, remaining_time):
        """Process final credit deduction for remaining session time"""
        try:
            if remaining_time <= 0:
                return
            
            cost_per_minute = credit_manager.calculate_total_cost_per_minute(self.selected_model)
            final_credits = cost_per_minute * remaining_time
            
            # Check and deduct final credits
            sufficient, current_balance, required = credit_manager.check_sufficient_credits(
                self.user_email, self.selected_model, remaining_time
            )
            
            if sufficient:
                # Deduct final credits with model info and disable auto-logging
                success, new_balance, message = credit_manager.deduct_credits(
                    self.user_email, final_credits, f"final_transcription_{self.selected_model}",
                    model=self.selected_model, auto_log=False
                )
                
                if success:
                    self.credits_deducted += final_credits
                    self.total_session_duration += remaining_time
                    
                    # Send final credit update
                    await self.websocket.send_json({
                        'type': 'session_finalized',
                        'final_credits_deducted': final_credits,
                        'total_credits_used': self.credits_deducted,
                        'final_balance': new_balance,
                        'total_duration_minutes': self.total_session_duration
                    })
                    
            
        except Exception as e:
            pass
    
    async def handle_insufficient_credits(self, current_balance, required_credits):
        """Handle when user runs out of credits during transcription"""
        try:
            # Stop transcription
            self.finish_event.set()
            
            # Send insufficient credits message
            await self.websocket.send_json({
                'type': 'credit_insufficient',
                'content': f'Credits exhausted during transcription. Session ended.',
                'error_code': 'CREDITS_EXHAUSTED',
                'current_balance': current_balance,
                'required_credits': required_credits,
                'session_duration': self.total_session_duration,
                'total_credits_used': self.credits_deducted
            })
            
            # Save final session data
            await self.finalize_session_with_credits()
            
            
        except Exception as e:
            pass

    async def _handle_quota_exceeded_error(self, text: str, error_message: str, timestamp=None):
        """Handle quota exceeded errors with prominent notification."""
        await self._safe_websocket_send({
            'type': 'translation_error',
            'content': f'Translation service out of credits. Transcription will continue but translations are disabled until credits are added.',
            'error_code': 'TRANSLATION_QUOTA_EXCEEDED',
            'severity': 'critical',
            'persistent': True,  # Frontend should show persistent notification
            'original_text': text[:30] + "..." if len(text) > 30 else text,
            'timestamp': timestamp
        })

    # connector to openai for keywords and summary
    async def generate_summary(self):
        """Generate summary of the full transcript using the selected provider."""
        if not self.Complete_Transcript.strip():
            print("No transcript available for summary")
            return

        prompt = self.system_prompt_summarizer.replace("{target_language}", self.target_language)
        input_text = f"{prompt}\n\nText:\n{self.Complete_Transcript.strip()}"

        if self.ai_provider == 'gemini':
            return await self.send_text_to_gemini(input_text)
        else:
            return await self.send_text_to_openai(input_text)

    async def extract_keywords(self):
        """Extract keywords from the full transcript using the selected provider."""
        if not self.Complete_Transcript.strip():
            print("No transcript available for keyword extraction")
            return

        input_text = f"{self.system_prompt_keywords.strip()}\n\nText:\n{self.Complete_Transcript.strip()}"

        if self.ai_provider == 'gemini':
            return await self.send_text_to_gemini(input_text)
        else:
            return await self.send_text_to_openai(input_text)

    # Conversation Handler
    def should_end_conversation(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.strip().lower()
        return re.search(r'\b(goodbye|bye)\b$', text) is not None

    def process_speaker_diarization(self, result):
        """Process speaker diarization information from Deepgram result."""
        try:
            transcript = result.channel.alternatives[0].transcript
            
            if not result.channel.alternatives[0].words:
                speaker_id = 0
                start_time = 0.0
                end_time = 0.0
            else:
                first_word = result.channel.alternatives[0].words[0]
                speaker_id = getattr(first_word, 'speaker', 0)  # Default to speaker 0 if no speaker info
                start_time = first_word.start
                
                # Get the last word for end time
                last_word = result.channel.alternatives[0].words[-1]
                end_time = last_word.end
            
            
            # Create speaker segment
            speaker_segment = {
                'speaker_id': speaker_id,
                'start_time': start_time,
                'end_time': end_time,
                'transcript': transcript,
                'is_final': result.is_final
            }

            # Initialize speaker transcript if not exists
            if speaker_id not in self.speaker_transcript:
                self.speaker_transcript[speaker_id] = ""
            
            # Add to speaker segments and update speaker transcript
            if result.is_final:
                self.speaker_segments.append(speaker_segment)
                # Clean up transcript before adding
                clean_transcript = transcript.strip()
                if clean_transcript:
                    self.speaker_transcript[speaker_id] += f" {clean_transcript}"
                
            return speaker_segment
            
        except Exception as e:
            return None

    # Real Time API
    async def connect_to_openai(self):
        try:
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

    async def connect_to_gemini(self):
        try:
            """Establish connection to Gemini's realtime API."""
            GEMINI_WS_URL = "wss://generativelanguage.googleapis.com/v1beta/models/gemini-pro:streamGenerateContent?alt=sse"

            headers = {
                "Authorization": f"Bearer {self.gemini_api_key}",
                "Content-Type": "application/json",
                "x-goog-api-key": self.gemini_api_key  # Depending on how the API key is passed
            }

            try:
                self.gemini_ws = await websockets.connect(
                    GEMINI_WS_URL,
                    additional_headers=headers
                )
            except TypeError:
                self.gemini_ws = await websockets.connect(
                    GEMINI_WS_URL,
                    extra_headers=headers
                )

            # You might need to send an initial payload like with OpenAI
            session_init_payload = {
                "type": "session.start",
                "model": self.selected_model or "gemini-pro",
                "instructions": self.system_prompt,
                "temperature": 0.6,
                "candidate_count": 1,
            }
            await self.gemini_ws.send(json.dumps(session_init_payload))
            # print("Connected to Gemini")
        except Exception as e:
            print(f"Error connecting to Gemini: {e}")
            raise e

    async def connect_to_claude(self):
        '''Claude websocket connection not available'''
        pass

    # Response Processing
    async def process_openai_responses(self):
        """Process responses from OpenAI's realtime API."""
        try:
            if self.openai_ws is None:
                print("OpenAI WebSocket is not connected")
                return
                
            async for message in self.openai_ws:
                try:
                    response = json.loads(message)
                    if response.get('type') == 'response.text.delta':
                        await self.websocket.send_json({
                            'type': 'assistant',
                            'content': response.get('delta'),
                        })

                    elif response.get('type') == 'response.text.done':
                        await self.websocket.send_json({
                            'type': 'assistant_done',
                            'content': 'Completed',
                        })
                except json.JSONDecodeError as e:
                    print(f"Error parsing OpenAI response: {e}")
                except Exception as e:
                    print(f"Error processing OpenAI message: {e}")
                
        except websockets.exceptions.ConnectionClosed:
            print("OpenAI connection closed")
            raise Exception('OpenAI connection closed')
        except Exception as e:
            print(f"Error processing OpenAI responses: {e}")
            raise e

    async def process_gemini_responses(self):
        """Process responses from Gemini's realtime API."""
        try:
            if self.gemini_ws is None:
                print("Gemini WebSocket is not connected")
                return

            async for message in self.gemini_ws:
                try:
                    response = json.loads(message)

                    if response.get("type") == "response.text.delta":
                        await self.websocket.send_json({
                            "type": "assistant",
                            "content": response.get("delta"),
                        })

                    elif response.get("type") == "response.text.done":
                        await self.websocket.send_json({
                            "type": "assistant_done",
                            "content": "Completed",
                        })

                except json.JSONDecodeError as e:
                    print(f"Error parsing Gemini response: {e}")
                except Exception as e:
                    print(f"Error processing Gemini message: {e}")

        except websockets.exceptions.ConnectionClosed:
            print("Gemini connection closed")
            raise Exception("Gemini connection closed")
        except Exception as e:
            print(f"Error processing Gemini responses: {e}")
            raise e

    async def process_claude_responses(self):
        '''Claude websocket connection not available'''
        pass

    # Send Message
    async def send_message_to_openai(self, text):
        """Send a message to OpenAI's realtime API."""
        try:
            if self.openai_ws is None:
                print("OpenAI WebSocket is not connected")
                return
                
            conversation_item = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": text
                        }
                    ]
                }
            }
            await self.openai_ws.send(json.dumps(conversation_item))
            await self.openai_ws.send(json.dumps({"type": "response.create"}))
        
        except Exception as e:
            print(f"Error sending to OpenAI: {e}")

    async def send_message_to_gemini(self, message):
        """Send user message to Gemini via realtime WebSocket."""
        if self.gemini_ws is None:
            print("Gemini WebSocket is not connected")
            return

        payload = {
            "type": "user.message",
            "content": message
        }

        try:
            await self.gemini_ws.send(json.dumps(payload))
        except Exception as e:
            print(f"Error sending message to Gemini: {e}")

    async def send_message_to_claude(self, message):
        '''Claude websocket connection not available'''
        pass

    # Text sending to AI models
    async def send_text_to_openai(self, input_text):
        """Send plain prompt text to OpenAI for one-shot completion."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.selected_model or "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ],
            "temperature": 0.7,
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=headers, json=data)
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"OpenAI summary/keyword error: {e}")
                return None
   
    async def send_text_to_gemini(self, input_text):
        """Send plain prompt text to Gemini for one-shot completion."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.gemini_api_key}"

        data = {
            "contents": [
                {
                    "parts": [
                        {"text": input_text}
                    ]
                }
            ]
        }

        headers = {
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=headers, json=data)
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                print(f"Gemini summary/keyword error: {e}")
                return None

    async def send_text_to_claude(self, input_text):
        '''Claude websocket connection not available'''
        pass

    # Deepgram Transcription
    async def transcribe_audio(self):
        if self.deepgram is None:
            print("Deepgram client not available, skipping audio transcription")
            return

        try:
            async def on_open(self, open, **kwargs):
                try:
                    if isinstance(open, bytes):
                        print(f"[Deepgram] on_open received bytes. Skipping. Length: {len(open)}")
                        return
                    print(f"Deepgram connection opened successfully")
                except Exception as e:
                    print(f"[Deepgram] Exception in on_open: {e} | type(open): {type(open)}")

            async def on_message(self_handler, result, **kwargs):
                try:
                    if self.finish_event.is_set():
                        print("[Deepgram] Finish event set, skipping message processing.")
                        return
                    if isinstance(result, bytes):
                        print(f"[Deepgram] Received bytes in on_message. Skipping. Length: {len(result)}")
                        return
                    if not hasattr(result, 'channel'):
                        print(f"[Deepgram] Received object without 'channel' attribute in on_message. Type: {type(result)}. Content: {str(result)[:200]}")
                        return
                    sentence = result.channel.alternatives[0].transcript
                    if len(sentence) == 0:
                        return
                    if getattr(result, 'is_final', False):
                        self.transcript_parts.append(sentence)
                        if self.stime == 0:
                            self.stime = result.channel.alternatives[0].words[0].start
                        if self.mode == 'speed':
                            await self.transcript_queue.put({'type': 'transcript_final', 'content': sentence, 'time': float(result.channel.alternatives[0].words[0].start)})
                        elif getattr(result, 'speech_final', False):
                            full_transcript = ' '.join(self.transcript_parts)
                            self.transcript_parts = []
                            self.stime = 0
                            await self.transcript_queue.put({'type': 'transcript_final', 'content': full_transcript, 'time': float(self.stime)})
                    else:
                        await self.transcript_queue.put({'type': 'transcript_interim', 'content': sentence, 'time': 0})
                except Exception as e:
                    print(f"Error processing transcription message: {e} | type(result): {type(result)} | content: {str(result)[:200]}")

            async def on_metadata(self, metadata, **kwargs):
                try:
                    if isinstance(metadata, bytes):
                        print(f"[Deepgram] on_metadata received bytes. Skipping. Length: {len(metadata)}")
                        return
                    print(f"Metadata received: {metadata}")
                except Exception as e:
                    print(f"[Deepgram] Exception in on_metadata: {e} | type(metadata): {type(metadata)}")

            async def on_speech_started(self, speech_started, **kwargs):
                try:
                    if isinstance(speech_started, bytes):
                        print(f"[Deepgram] on_speech_started received bytes. Skipping. Length: {len(speech_started)}")
                        return
                    print(f"Speech Started")
                except Exception as e:
                    print(f"[Deepgram] Exception in on_speech_started: {e} | type(speech_started): {type(speech_started)}")

            async def on_utterance_end(self_handler, utterance_end, **kwargs):
                try:
                    if isinstance(utterance_end, bytes):
                        print(f"[Deepgram] on_utterance_end received bytes. Skipping. Length: {len(utterance_end)}")
                        return
                    if self.mode != 'speed' and len(self.transcript_parts) > 0:
                        full_transcript = ' '.join(self.transcript_parts)
                        self.transcript_parts = []
                        await self.transcript_queue.put({'type': 'transcript_final', 'content': full_transcript})
                except Exception as e:
                    print(f"Error processing utterance end: {e}")

            async def on_close(self, close, **kwargs):
                try:
                    if isinstance(close, bytes):
                        print(f"[Deepgram] on_close received bytes. Skipping. Length: {len(close)}")
                        return
                    print(f"Deepgram connection closed: {close}")
                    if not self.finish_event.is_set():
                        print("Unexpected Deepgram disconnection, attempting to reconnect...")
                        try:
                            await asyncio.sleep(1)  # Brief delay before reconnect
                            await self.transcribe_audio()  # Attempt to reconnect
                        except Exception as e:
                            print(f"Reconnection attempt failed: {e}")
                except Exception as e:
                    print(f"[Deepgram] Exception in on_close: {e} | type(close): {type(close)}")

            async def on_error(self, error, **kwargs):
                try:
                    if isinstance(error, bytes):
                        print(f"[Deepgram] on_error received bytes. Skipping. Length: {len(error)}")
                        return
                    print(f"Deepgram error: {error}")
                    if not self.finish_event.is_set():
                        await self.websocket.send_json({
                            'type': 'transcription_error',
                            'content': f'Transcription error occurred: {error}',
                            'error_code': 'DEEPGRAM_ERROR'
                        })
                except Exception as e:
                    print(f"[Deepgram] Exception in on_error: {e} | type(error): {type(error)}")

            async def on_unhandled(self, unhandled, **kwargs):
                try:
                    if isinstance(unhandled, bytes):
                        print(f"[Deepgram] on_unhandled received bytes. Skipping. Length: {len(unhandled)}")
                        return
                    print(f"Unhandled Deepgram message: {unhandled}")
                except Exception as e:
                    print(f"[Deepgram] Exception in on_unhandled: {e} | type(unhandled): {type(unhandled)}")

            # Use the new asyncwebsocket instead of deprecated asynclive
            try:
                dg_connection = self.deepgram.listen.asyncwebsocket.v('1')
            except AttributeError:
                # Fallback for older SDK versions
                print("Warning: Using deprecated asynclive API")
                dg_connection = self.deepgram.listen.asynclive.v('1')
            
            dg_connection.on(LiveTranscriptionEvents.Open, on_open)
            dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
            dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
            dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)
            dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
            dg_connection.on(LiveTranscriptionEvents.Close, on_close)
            dg_connection.on(LiveTranscriptionEvents.Error, on_error)
            dg_connection.on(LiveTranscriptionEvents.Unhandled, on_unhandled)

            connection_result = await dg_connection.start(self.dg_connection_options)
            if connection_result is False:
                print("Failed to start Deepgram connection")
                print('*'*100)
                print(f"Deepgram connection error: {connection_result}")
                print("This is likely due to:")
                print("1. Invalid API key")
                print("2. Insufficient credits (even if dashboard shows credits)")
                print("3. API key permissions issue")
                print("4. Account billing status")
                print("Solutions to try:")
                print("- Regenerate your API key at https://console.deepgram.com/")
                print("- Contact Deepgram support if credits show but API fails")
                print("- Check if your account has any restrictions")
                print('*'*100)
                raise Exception('Failed to connect to Deepgram')

            try:
                while not self.finish_event.is_set():
                    try:
                        # Add timeout to prevent hanging
                        data = await asyncio.wait_for(self.websocket.receive_bytes(), timeout=30.0)
                        if data:  # Check if data is not empty
                            await dg_connection.send(data)
                        else:
                            print("Received empty data from WebSocket")
                    except asyncio.TimeoutError:
                        if not self.finish_event.is_set():
                            continue  # Keep listening if session is still active
                        break  # Break if session is finished
                    except Exception as e:
                        print(f"Error receiving/sending data: {e}")
                        if not self.finish_event.is_set():
                            await asyncio.sleep(0.1)  # Brief pause before retry
                            continue
                        break
                    # Break the loop if the connection is closed
                    if self.finish_event.is_set():
                        print("[Deepgram] Finish event set in main loop, breaking.")
                        break
                print("[Deepgram] Main loop exited")
            except Exception as e:
                print(f"Error in transcription loop: {e}")
                raise
            finally:
                print("Closing Deepgram connection...")
                try:
                    await dg_connection.finish()
                except Exception as e:
                    print(f"Error closing Deepgram connection: {e}")

        except Exception as e:
            print(f"[Deepgram] OUTER EXCEPTION: {e} | type: {type(e)}")
            import traceback
            traceback.print_exc()
            if not self.finish_event.is_set():
                await self.websocket.send_json({
                    'type': 'transcription_error',
                    'content': f'Transcription service encountered an error: {e}. Please try again.',
                    'error_code': 'DEEPGRAM_ERROR'
                })
            # Don't raise the exception, just log it to prevent the whole system from crashing
            print("Continuing without Deepgram transcription...")


    # Session Management
    async def initialize_session(self):
        """Initialize the session with credit tracking and session setup."""
        try:
            if self.session_initialized:
                return True
                
            # Initialize credit tracking
            if not await self.initialize_credit_tracking():
                return False
            
            # Start credit monitoring for paid users
            await self.start_credit_monitoring()
            
            # Initialize session timestamp
            self.Object_session["Session_Start_Time"] = datetime.now().isoformat()
            
            # Send session initialization confirmation
            await self.websocket.send_json({
                'type': 'session_initialized',
                'content': 'Session started successfully',
                'user_email': self.user_email,
                'is_paid_user': self.is_paid_user(),
                'model': self.selected_model,
                'ai_mode': self.AI_mode
            })
            
            self.session_initialized = True
            return True
            
        except Exception as e:
            print(f"Error initializing session: {e}")
            return False

    async def update_session_data(self, transcript_data):
        """Update session data with new transcript information."""
        try:
            if transcript_data['type'] == 'transcript_final':
                # Add to complete transcript
                self.Complete_Transcript += f" {transcript_data['content']}"
                self.Object_session["Transcript"] = self.Complete_Transcript.strip()
                
                # Update last activity for keep-alive
                self.last_activity = datetime.now()
                
        except Exception as e:
            print(f"Error updating session data: {e}")

    async def update_translation_data(self, translation_content):
        """Update session data with translation information."""
        try:
            self.Complete_Translation += f" {translation_content}"
            self.Object_session["Translation"] = self.Complete_Translation.strip()
            
        except Exception as e:
            print(f"Error updating translation data: {e}")

    async def finalize_session(self):
        """Finalize the session with all cleanup tasks."""
        try:
            if self.cleanup_completed:
                return
                
            print("Finalizing session...")
            
            # Set session end time
            self.Object_session["Session_End_Time"] = datetime.now().isoformat()
            
            # Stop the finish event to halt all tasks
            self.finish_event.set()
            
            # Cancel credit monitoring task
            if self.credit_task:
                self.credit_task.cancel()
                try:
                    await self.credit_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel keep-alive task
            if self.keep_alive_task:
                self.keep_alive_task.cancel()
                try:
                    await self.keep_alive_task
                except asyncio.CancelledError:
                    pass
            
            # Finalize credits for paid users
            await self.finalize_session_with_credits()
            
            # Save session to database
            session_saved = await self.save_session_to_database()
            
            # Send final session summary
            await self.websocket.send_json({
                'type': 'session_finalized',
                'content': 'Session completed successfully',
                'session_saved': session_saved,
                'total_credits_used': self.credits_deducted,
                'session_duration_minutes': self.total_session_duration,
                'transcript_length': len(self.Complete_Transcript),
                'translation_length': len(self.Complete_Translation),
                'timestamp': datetime.now().isoformat()
            })
            
            self.cleanup_completed = True
            print("Session finalization completed")
            
        except Exception as e:
            print(f"Error finalizing session: {e}")

    async def handle_websocket_disconnect(self):
        """Handle WebSocket disconnection gracefully."""
        try:
            print("WebSocket disconnected, cleaning up...")
            await self.finalize_session()
            
        except Exception as e:
            print(f"Error handling WebSocket disconnect: {e}")

    # Database Management
    async def Premium_Feature(self):
        self.Object_session["Keywords"] = await self.extract_keywords()
        self.Object_session["Summary"] = await self.generate_summary() 

    async def save_session_to_database(self):
        """Enhanced save method with better error handling and data validation."""
        try:
            if not self.user_email:
                print("No user email provided, skipping database save")
                return False
            
            # Prevent duplicate saves
            if self.session_saved:
                print("Session already saved, skipping duplicate save")
                return True
            
            # Validate session content
            transcript_content = self.Object_session.get("Transcript", "").strip()
            translation_content = self.Object_session.get("Translation", "").strip()
            
            # Don't save empty sessions
            if not transcript_content or len(transcript_content) < 10:
                print("Session content too short, not saving to database")
                return False
            
            # Generate premium features for paid users
            if self.is_paid_user() and self.AI_mode:
                print("Generating premium features for paid user...")
                await self.Premium_Feature()
            
            # Prepare session data
            session_data = {
                "Original Text": transcript_content,
                "Translated Text": translation_content,
                "Summary": self.Object_session.get("Summary", ""),
                "Original Language": self.languages[0],
                "Translated Language": self.languages[1],
                "Keywords": self.Object_session.get("Keywords", ""),
                "Model_Used": self.selected_model,
                "Credits_Used": self.credits_deducted,
                "Session_Duration_Minutes": self.total_session_duration,
                "AI_Mode": self.AI_mode,
                "AI_Provider": self.ai_provider if self.AI_mode else None,
                "Session_Start_Time": self.Object_session.get("Session_Start_Time"),
                "Session_End_Time": self.Object_session.get("Session_End_Time"),
                "Timestamp": datetime.now().isoformat()
            }
            
            # Find user in database
            user = Find_User_DB(self.user_email)
            if not user:
                print(f"User not found in database: {self.user_email}")
                return False
            
            # Initialize session structure
            current_date = self.date
            current_time = self.time
            
            if "Session" not in user:
                user["Session"] = {}
            if current_date not in user["Session"]:
                user["Session"][current_date] = {}
            if self.web_page_name not in user["Session"][current_date]:
                user["Session"][current_date][self.web_page_name] = {}
            
            # Save session data
            user["Session"][current_date][self.web_page_name][current_time] = session_data
            
            # Update database
            result = collection.update_one(
                {"Email": self.user_email}, 
                {"$set": {"Session": user["Session"]}}
            )
            
            if result.modified_count > 0:
                self.session_saved = True
                print(f"Session saved successfully for user: {self.user_email}")
                return True
            else:
                print("Database update failed - no documents modified")
                return False
                
        except Exception as e:
            print(f"Error saving session to database: {e}")
            return False
  
    # Manage Conversation Handler
    async def manage_conversation(self):
        """Enhanced conversation management with session tracking."""
        while not self.finish_event.is_set():
            try:
                # Add timeout to prevent hanging
                transcript = await asyncio.wait_for(
                    self.transcript_queue.get(), 
                    timeout=30.0
                )
                
                await self.websocket.send_json(transcript)
                
                # Update session data
                await self.update_session_data(transcript)

                if transcript['type'] == 'transcript_final':
                    # Send to appropriate AI provider
                    if self.AI_mode:
                        if self.ai_provider == 'gemini':
                            await self.send_message_to_gemini(transcript['content'])
                        else:
                            await self.send_message_to_openai(transcript['content'])
                    
                    # Check if conversation should end
                    if self.should_end_conversation(transcript['content']):
                        print("Conversation end detected")
                        break

            except asyncio.TimeoutError:
                # Check if session is still active
                if not self.finish_event.is_set():
                    continue
                else:
                    break
            except Exception as e:
                print(f'Error in Managing Conversations: {e}')
                break

    # Main Function
    async def run(self):
        """Enhanced main run method with proper session management."""
        try:
            print("Starting Assistant session...")
            
            # Initialize session
            if not await self.initialize_session():
                print("Failed to initialize session")
                return
            
            # Connect to the appropriate AI provider
            if self.AI_mode:
                if self.ai_provider == 'gemini':
                    await self.connect_to_gemini()
                else:
                    await self.connect_to_openai()
            
            print("AI connections established, starting main tasks...")
            
            # Create main task group
            async with asyncio.TaskGroup() as tg:
                # Core transcription task
                transcription_task = tg.create_task(self.transcribe_audio())
                
                # Conversation management task
                conversation_task = tg.create_task(self.manage_conversation())
                
                # AI response processing task (only if AI mode is enabled)
                if self.AI_mode:
                    if self.ai_provider == 'gemini':
                        ai_task = tg.create_task(self.process_gemini_responses())
                    else:
                        ai_task = tg.create_task(self.process_openai_responses())
                
                # Optional: Add keep-alive task for long sessions
                if self.keep_alive_task:
                    keep_alive_task = tg.create_task(self.keep_alive_task)
                
                print("All tasks started, waiting for completion...")
                
        except Exception as eg:
            # Handle multiple exceptions from TaskGroup
            print(f"TaskGroup exceptions occurred:")
            for exc in eg.exceptions:
                print(f"  - {type(exc).__name__}: {exc}")
                
                # Handle specific exceptions
                if isinstance(exc, WebSocketDisconnect):
                    print("WebSocket disconnected")
                    await self.handle_websocket_disconnect()
                elif isinstance(exc, asyncio.CancelledError):
                    print("Task was cancelled")
                else:
                    print(f"Unexpected error: {exc}")
                    
        except Exception as e:
            print(f"Unexpected error in run method: {e}")
            
        finally:
            print("Cleaning up session...")
            try:
                # Ensure session finalization
                await self.finalize_session()
                
                # Close WebSocket connections
                if self.openai_ws:
                    await self.openai_ws.close()
                if self.gemini_ws:
                    await self.gemini_ws.close()
                
                # Close client WebSocket if still connected
                if (self.websocket and 
                    self.websocket.client_state != WebSocketState.DISCONNECTED):
                    await self.websocket.close()
                    
                print("Session cleanup completed")
                
            except Exception as cleanup_error:
                print(f"Error during cleanup: {cleanup_error}")
