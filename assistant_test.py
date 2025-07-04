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



MODEL_BEING_USED='gpt-4o-realtime-preview'

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
                                    
                                    # These two are required
                                    encoding="linear16",         # Use "linear16" for 16-bit PCM
                                    sample_rate=16000            # or match whatever your audio input uses
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
            self.selected_model = MODEL_BEING_USED
            #logger.info(f"Speed mode detected: forcing model to {CHAT_GPT_MODEL_BEING_USED} for optimal performance")
        elif not selected_model:
            # Default fallback model if none selected
            self.selected_model = MODEL_BEING_USED
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
                print("Warning: Using deprecated asynclive API")
                dg_connection = self.deepgram.listen.asynclive.v('1')
            
            # Set up event handlers
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
