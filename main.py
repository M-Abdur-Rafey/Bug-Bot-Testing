import asyncio
import os
from fastapi import FastAPI, WebSocket, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from assistant import Assistant
from dotenv import load_dotenv
from google.auth.transport import requests
from google.oauth2 import id_token
import json
from User import User
import jwt
from datetime import datetime, timedelta
import logging
from jwt.exceptions import PyJWTError
from Autentication import get_User_API_Key, check_openai_key, Find_User_DB
from assistant import CHAT_GPT_MODEL_BEING_USED
from payment import credit_manager, Credit_cost_per_minute
from Packages import Packages
from Stripe import get_stripe_manager

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'https://gpt-live-translator.techquest.ai',
        'http://localhost:5173',
        'chrome-extension://*',
        'moz-extension://*',
        "https://accounts.google.com", 
        "https://www.googleapis.com",   
    ],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
    allow_origin_regex=r"^chrome-extension://.*|^moz-extension://.*"  # More flexible extension origin handling
)

load_dotenv()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 3

def create_jwt_token(user_data: dict) -> str:
    payload = {
        "email": user_data.get("email"),
        "name": user_data.get("name"),
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.utcnow(),
        "sub": user_data.get("email"),
    }
    
    payload = {k: v for k, v in payload.items() if v is not None}
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token

def verify_jwt_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except PyJWTError:
        return None

async def safe_websocket_close(websocket: WebSocket, code: int = 1000, reason: str = "Connection closed"):
    """Safely close a WebSocket connection without raising exceptions if already closed."""
    try:
        if websocket.client_state.value <= 2:  # Check if WebSocket is still open
            await websocket.close(code=code, reason=reason)
    except Exception as e:
        #logger.error(f"Error closing websocket: {e}")
        pass

async def safe_websocket_send(websocket: WebSocket, message: dict):
    """Safely send a message to WebSocket without raising exceptions if already closed."""
    try:
        if websocket.client_state.value <= 2:  # Check if WebSocket is still open
            await websocket.send_json(message)
            return True
    except Exception as e:
        #logger.debug(f"WebSocket already closed or error during send: {str(e)}")
        return False

@app.head('/health')
@app.get('/health')
def health_check():
    return 'ok'

@app.get('/get-packages')
async def get_packages():
    return Packages

@app.post('/register')
async def register(request: Request, response: Response):
    try:
        data = await request.json()
        data = {key.lower(): value for key, value in data.items()}
        
        required_fields = ['name', 'email', 'password']
        for field in required_fields:
            if not data.get(field):
                raise HTTPException(status_code=400, detail=f"{field} is required")
        
        #logger.info(f"Data: {data}")
        user = User()
        
        #logger.info(f"Registering user: {data}")
        if user.Add_User(data):
            # Get the created user data to check for API keys
            email = data.get('email').lower()
            #logger.info(f"Email: {email}")
            user_data = user.Get_User_Data(email)

            #logger.info(f"User data: {user_data}")
            
            if user_data:
                safe_user_data = {
                    'email': user_data.get('Email', '').lower(),
                    'name': user_data.get('Name'),
                    'created_at': user_data.get('Created_At'),
                    'api_keys': user_data.get('API_Key', {}),
                    'sessions': user_data.get('Session', {}),
                    'summary_and_keywords': user_data.get('Summary_and_Keywords', 0)
                }
                
                token = create_jwt_token({
                    'email': user_data.get('Email', '').lower(),
                    'name': user_data.get('Name')
                })
                
                # Set cookie for cross-tab access
                response.set_cookie(
                    key="access_token",
                    value=token,
                    httponly=True,
                    samesite="lax",
                    secure=False,
                    max_age=86400 * 7,
                    path="/"
                )
                
                # Check if user has API keys
                has_api_keys = check_user_has_api_keys(email)
                
                return {
                    'success': True,
                    'message': 'Registered successfully',
                    'user': safe_user_data,
                    'token': token,
                    'token_type': 'bearer',
                    'has_api_keys': has_api_keys
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to retrieve user data after registration")
        else:
            raise HTTPException(status_code=409, detail="User already exists")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during registration")

@app.post('/login')
async def login(request: Request, response: Response):
    try:
        data = await request.json()
        data = {key.lower(): value for key, value in data.items()}
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            raise HTTPException(status_code=400, detail="Email and password are required")
        
        email = email.lower()
        user = User()
        
        if user.Find_User(email):
            if user.Check_User_Credentials(email, password):
                user_data = user.Get_User_Data(email)
                if user_data:
                    safe_user_data = {
                        'email': user_data.get('Email', '').lower(),
                        'name': user_data.get('Name'),
                        'created_at': user_data.get('Created_At'),
                        # 'api_keys': user_data.get('API_Key', {}),
                        'sessions': user_data.get('Session', {}),
                        'summary_and_keywords': user_data.get('Summary_and_Keywords', 0)
                    }
                    
                    token = create_jwt_token({
                        'email': user_data.get('Email', '').lower(),
                        'name': user_data.get('Name')
                    })
                    
                    response.set_cookie(
                        key="access_token",
                        value=token,
                        httponly=True,
                        samesite="lax",
                        secure=False,
                        max_age=86400 * 7,
                        path="/"
                    )
                    
                    # Check if user has API keys
                    has_api_keys = check_user_has_api_keys(email)
                    print(datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS))
                    print(datetime.utcnow())
                    return {
                        'success': True,
                        'message': 'Login successful',
                        'user': safe_user_data,
                        'token': token,
                        'token_type': 'bearer',
                        # 'has_api_keys': has_api_keys,
                        'token_expiry_time': (datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)).isoformat() + "Z",
                        'token_created_time': datetime.utcnow().isoformat() + "Z"
                    }
                else:
                    raise HTTPException(status_code=500, detail="Failed to retrieve user data")
            else:
                raise HTTPException(status_code=401, detail="Invalid email or password")
        else:
            raise HTTPException(status_code=401, detail="Invalid email or password")
    except HTTPException:
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal server error during login")

@app.post('/auth/google')
async def google_auth(request: Request, response: Response):
    try:
        data = await request.json()
        credential = data.get('credential')
        is_sign_in = data.get('isSignIn', False)
        
        if not credential:
            raise HTTPException(status_code=400, detail="No credential provided")
        
        google_client_id = os.getenv("GOOGLE_CLIENT_ID", "VITE_GOOGLE_CLIENT_ID")
        
        try:
            idinfo = id_token.verify_oauth2_token(
                credential, requests.Request(), google_client_id
            )
            
            user_id = idinfo['sub']
            email = idinfo['email']
            name = idinfo.get('name', '')
            picture = idinfo.get('picture', '')
            
            # Check if user exists or create new user
            user = User()
            email = email.lower()
            
            if not user.Find_User(email):
                # Create new user for Google registration
                user_creation_data = {
                    'name': name,
                    'email': email,
                    'password': user_id,  # Using Google ID as password for OAuth users
                    'api_key': {}  # Empty API keys initially
                }
                user.Add_User(user_creation_data)
            
            # Get user data from database
            user_data = user.Get_User_Data(email)
            
            if user_data:
                safe_user_data = {
                    'email': user_data.get('Email', '').lower(),
                    'name': user_data.get('Name'),
                    'created_at': user_data.get('Created_At'),
                    'api_keys': user_data.get('API_Key', {}),
                    'sessions': user_data.get('Session', {}),
                    'summary_and_keywords': user_data.get('Summary_and_Keywords', 0),
                    'google_id': user_id,
                    'picture': picture
                }
                
                # Create token with timing information
                current_time = datetime.utcnow()
                expiry_time = current_time + timedelta(days=7)  # 7 days expiry
                
                token = create_jwt_token({
                    'email': email,
                    'name': name,
                    'exp': expiry_time
                })
                
                # Set cookie for cross-tab access
                response.set_cookie(
                    key="access_token",
                    value=token,
                    httponly=True,
                    samesite="lax",
                    secure=False,
                    max_age=86400 * 7,
                    path="/"
                )
                
                # Check if user has API keys
                has_api_keys = check_user_has_api_keys(email)
                
                return {
                    'success': True,
                    'message': f'Google {"sign-in" if is_sign_in else "registration"} successful',
                    'user': safe_user_data,
                    'token': token,
                    'token_type': 'bearer',
                    'token_created_time': current_time.isoformat(),
                    'token_expiry_time': expiry_time.isoformat(),
                    'has_api_keys': has_api_keys
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to retrieve user data")
            
        except ValueError as e:
            raise HTTPException(status_code=401, detail=f"Invalid Google token: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Google authentication failed")

@app.post('/logout')
async def logout(response: Response):
    try:
        response.delete_cookie(
            key="access_token",
            path="/",
            samesite="lax"
        )
        return {"success": True, "message": "Logged out successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during logout")
    
@app.post('/verify-token')
async def verify_token(request: Request):
    try:
        data = await request.json()
        token = data.get('token')
        
        if not token:
            raise HTTPException(status_code=400, detail="Token is required")
        
        if token.startswith('Bearer '):
            token = token[7:]
        
        payload = verify_jwt_token(token)
        
        return {
            'success': True,
            'message': 'Token is valid',
            'user': {
                'email': payload.get('email'),
                'name': payload.get('name'),
                'exp': payload.get('exp'),
                'iat': payload.get('iat')
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Token verification failed")

@app.get('/validate-token')
async def validate_token(request: Request):
    try:
        # Get token from Authorization header
        authorization = request.headers.get('Authorization')
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header is required")
        
        token = authorization
        if token.startswith('Bearer '):
            token = token[7:]
        
        payload = verify_jwt_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Check if token is expired
        exp_timestamp = payload.get('exp')
        if exp_timestamp:
            import time
            current_time = time.time()
            if current_time > exp_timestamp:
                raise HTTPException(status_code=401, detail="Token has expired")
        
        return {
            'success': True,
            'message': 'Token is valid',
            'user': {
                'email': payload.get('email'),
                'name': payload.get('name'),
                'exp': payload.get('exp'),
                'iat': payload.get('iat')
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail="Token validation failed")

@app.get('/get-dates')
async def get_data(request: Request):
    try:
        # Get token from Authorization header
        authorization = request.headers.get('Authorization')
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header is required")
        
        token = authorization
        if token.startswith('Bearer '):
            token = token[7:]

        payload = verify_jwt_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = payload.get('email')
        user = User()
        user_data = user.Get_User_Dates(email)
        
        if user_data:
            return {
                'success': True,
                'message': 'Data retrieved successfully',
                'dates': user_data
            }
        else:
            raise HTTPException(status_code=404, detail="User not found")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during get data")

@app.get('/get-data-by-date')
async def get_data_by_date(request: Request):
    try:
        # Get token from Authorization header
        authorization = request.headers.get('Authorization')
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header is required")
        
        token = authorization
        if token.startswith('Bearer '):
            token = token[7:]
        
        payload = verify_jwt_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = payload.get('email')
        date = request.query_params.get('date')  # Get date from query parameters
        
        if not date:
            raise HTTPException(status_code=400, detail="Date parameter is required")
        
        user = User()
        user_data = user.Get_User_Data_By_Date(email, date)
        
        if user_data:
            return {
                'success': True,
                'message': 'Data retrieved successfully',
                'data': user_data
            }
        else:
            raise HTTPException(status_code=404, detail="User not found")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during get data by date")


@app.get('/get-model-cost')
async def get_model_cost(request: Request):
    from payment import Model_table
    return Model_table

@app.post('/update-API-Key')
async def update_API_Key(request: Request):
    try:
        data = await request.json()
        token = data.get('token')
        api_key = data.get('api_key')
        summary_and_keywords = data.get('summary_and_keywords')

        payload = verify_jwt_token(token)

        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = payload.get('email')
        user = User()
        
        # check if the api key is valid by pinging the openai api
        if check_openai_key(api_key['OpenAI']) == False:
            raise HTTPException(status_code=400, detail="Invalid OpenAI API key")
        
        # Update API key
        api_key_result = user.Update_API_Key(email, api_key)
        if not api_key_result:
            #logger.error(f"Failed to update API key for user: {email}")
            raise HTTPException(status_code=500, detail="Failed to update API key")
        
        # Update summary and keywords preference
        s_and_k_result = user.Update_S_and_K(email, summary_and_keywords)
        if not s_and_k_result:
            #logger.error(f"Failed to update Summary and Keywords preference for user: {email}")
            raise HTTPException(status_code=500, detail="Failed to update Summary and Keywords preference")
        
        #logger.info(f"Successfully updated API key and preferences for user: {email}")
        return {
            'success': True,
            'message': 'API key updated successfully'
        }
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Unexpected error during update API key: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error during update API key: {str(e)}")

@app.post('/get-api-key')
async def get_api_key(request: Request):
    try:
        data = await request.json()
        token = data.get('token')
        
        payload = verify_jwt_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = payload.get('email')
        user = User()
        user_data = user.Get_User_Data(email)
        api_key = user_data.get('API_Key', {})
        Summary_and_Keywords = user_data.get('Summary_and_Keywords')
        paid_user = user_data.get('Paid_User', False)

        # first is deepgram then openai then claude
        To_return = {}
        if api_key.get('Deepgram'):
            To_return['Deepgram'] = api_key.get('Deepgram')
        if api_key.get('OpenAI'):
            To_return['OpenAI'] = api_key.get('OpenAI')
        if api_key.get('Claude'):
            To_return['Claude'] = api_key.get('Claude')
        
        return {
            'success': True,
            'message': 'Settings retrieved successfully',
            'api_key': To_return,
            'summary_and_keywords': Summary_and_Keywords,
            'paid_user': paid_user
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during get API key")
    
@app.post('/Check-Payment-Status')
async def check_payment_status(request: Request):
    try:
        data = await request.json()
        token = data.get('token')
        
        payload = verify_jwt_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = payload.get('email')
        user = User()
        user_data = user.Get_User_Data(email)
        paid_user = user_data.get('Paid_User', False)
        
        if paid_user:
            return {
                'success': True,
                'message': 'Payment status retrieved successfully',
                'paid_user': paid_user
            }
        return {
            'success': False,
            'message': 'Payment status retrieved successfully',
            'paid_user': False
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during check payment status")

@app.post('/verify-payment')
async def verify_payment(request: Request):
    """
    Verify a payment and add credits to user account with idempotency protection.
    This endpoint ensures that credits are only added once per payment session.
    """
    try:
        # Extract and verify token
        token = None
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            token = request.cookies.get("access_token")
        
        if not token:
            raise HTTPException(status_code=401, detail="Token required")
        
        user_data = verify_jwt_token(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = user_data.get('email', '').lower()
        
        # Get request data
        data = await request.json()
        session_id = data.get('session_id')
        payment_intent_id = data.get('payment_intent')
        
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        
        #logger.info(f"Verifying payment for user: {email}, session: {session_id}")
        
        # First, check if this payment has already been processed
        from Autentication import collection
        existing_user = collection.find_one({"Email": email})
        
        if existing_user:
            credit_transactions = existing_user.get('CreditTransactions', [])
            
            # Check if we already have a transaction for this session ID
            existing_transaction = None
            for transaction in credit_transactions:
                if (transaction.get('stripe_session_id') == session_id or 
                    transaction.get('stripe_payment_intent_id') == payment_intent_id):
                    existing_transaction = transaction
                    break
            
            if existing_transaction:
                #logger.info(f"Payment already processed for session {session_id}")
                
                # Return the existing payment details
                payment_details = {
                    "transaction_id": existing_transaction.get('stripe_session_id', session_id),
                    "amount": str(existing_transaction.get('package_price', 0)),
                    "credits": str(int(existing_transaction.get('amount', 0))),
                    "currency": existing_transaction.get('payment_currency', 'USD'),
                    "payment_method": "card",
                    "date": existing_transaction.get('timestamp', datetime.utcnow().isoformat()),
                    "status": "completed"
                }
                
                return {
                    "success": True,
                    "already_processed": True,
                    "payment_details": payment_details,
                    "message": "Payment has already been processed"
                }
        
        # Payment hasn't been processed yet, verify with Stripe and process
        try:
            stripe_manager = get_stripe_manager()
            import stripe
            
            # Try to retrieve the checkout session first
            try:
                session = stripe.checkout.Session.retrieve(session_id)
                
                if session.payment_status != 'paid':
                    raise HTTPException(status_code=400, detail="Payment not completed in Stripe")
                
                # Extract payment information from session
                metadata = session.metadata or {}
                customer_email = metadata.get('customer_email', email)
                package_name = metadata.get('package_name', 'Unknown')
                credits_str = metadata.get('credits', '0')
                
                try:
                    credits = int(credits_str)
                except:
                    credits = 0
                
                package_price = session.amount_total / 100.0 if session.amount_total else 0.0
                payment_intent_id = session.payment_intent
                
            except stripe.error.StripeError as e:
                # If session retrieval fails, try with payment intent
                if payment_intent_id:
                    payment_intent = stripe.PaymentIntent.retrieve(payment_intent_id)
                    
                    if payment_intent.status != 'succeeded':
                        raise HTTPException(status_code=400, detail="Payment not completed")
                    
                    # Extract info from payment intent metadata
                    metadata = payment_intent.metadata or {}
                    credits_str = metadata.get('credits', '0')
                    credits = int(credits_str) if credits_str.isdigit() else 0
                    package_price = payment_intent.amount / 100.0
                    package_name = metadata.get('package_type', 'Unknown')
                    customer_email = email
                else:
                    raise HTTPException(status_code=400, detail=f"Unable to verify payment: {str(e)}")
            
            # Add credits to user account
            success, new_balance, message = credit_manager.add_credits(
                customer_email,
                credits,
                f"Stripe payment - {package_name} package (Session: {session_id})",
                auto_log=False
            )
            
            if not success:
                raise HTTPException(status_code=500, detail=f"Failed to add credits: {message}")
            
            # Create transaction record
            from datetime import datetime
            transaction_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": "purchase",
                "amount": float(credits),
                "stripe_session_id": session_id,
                "stripe_payment_intent_id": payment_intent_id,
                "package_name": package_name,
                "package_price": package_price,
                "payment_currency": "USD",
                "payment_method": "stripe",
                "payment_status": "completed",
                "invoice_data": {
                    "transaction_id": session_id,
                    "customer_email": customer_email,
                    "amount_paid": package_price,
                    "credits_purchased": credits,
                    "package_description": f"{package_name} Package - {credits} Credits",
                    "payment_date": datetime.utcnow().isoformat()
                }
            }
            
            # Add transaction to user's record
            collection.update_one(
                {"Email": customer_email},
                {"$push": {"CreditTransactions": transaction_data}},
                upsert=False
            )
            
            #logger.info(f"Successfully processed payment: Added {credits} credits to {customer_email}")
            
            # Return payment details
            payment_details = {
                "transaction_id": session_id,
                "amount": str(package_price),
                "credits": str(credits),
                "currency": "USD",
                "payment_method": "card",
                "date": transaction_data["timestamp"],
                "status": "completed"
            }
            
            return {
                "success": True,
                "newly_processed": True,
                "payment_details": payment_details,
                "new_balance": new_balance,
                "message": f"Successfully added {credits} credits to your account"
            }
            
        except stripe.error.StripeError as e:
            #logger.error(f"Stripe error during payment verification: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Payment verification failed: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Error in payment verification: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to verify payment")

@app.post('/update-paid-user')
async def update_paid_user(request: Request):
    try:
        data = await request.json()
        token = data.get('token')
        paid_user_status = data.get('paid_user')
        
        if paid_user_status is None:
            raise HTTPException(status_code=400, detail="paid_user status is required")
        
        payload = verify_jwt_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = payload.get('email')
        user = User()
        
        # Update paid user status
        result = user.Update_Paid_User(email, paid_user_status)
        if not result:
            #logger.error(f"Failed to update paid user status for user: {email}")
            raise HTTPException(status_code=500, detail="Failed to update subscription status")
        
        #logger.info(f"Successfully updated paid user status for user: {email} to {paid_user_status}")
        return {
            'success': True,
            'message': 'Subscription status updated successfully',
            'paid_user': paid_user_status
        }
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Unexpected error during update paid user status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error during update subscription status: {str(e)}")

@app.get('/get-paid-user-status')
async def get_paid_user_status(request: Request):
    """
    Returns the paid_user status for the authenticated user.
    Token must be provided in the Authorization header as Bearer token.
    """
    try:
        # Get token from Authorization header
        authorization = request.headers.get('Authorization')
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header is required")
        
        token = authorization
        if token.startswith('Bearer '):
            token = token[7:]
        
        payload = verify_jwt_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = payload.get('email')
        user = User()
        user_data = user.Get_User_Data(email)
        paid_user = user_data.get('Paid_User', False)
        
        return {
            'success': True,
            'paid_user': paid_user
        }
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Error getting paid user status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get paid user status")
    
@app.get('/Get-User-Data')
async def get_user_data(request: Request):
    try:
        # Get token from Authorization header
        authorization = request.headers.get('Authorization')
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header is required")
        
        token = authorization
        if token.startswith('Bearer '):
            token = token[7:]
        
        payload = verify_jwt_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = payload.get('email')
        user = User()
        user_data = user.Get_User_Data(email)
        
        if user_data:
            return {
                'success': True,
                'message': 'User data retrieved successfully',
                'data': user_data
            }
        else:
            raise HTTPException(status_code=404, detail="User not found")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during get user data")

@app.post('/set-ai-provider')
async def set_ai_provider(request: Request):
    try:
        data = await request.json()
        token = data.get('token')
        ai_provider = data.get('ai_provider', 'openai').lower()
        
        payload = verify_jwt_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        if ai_provider not in ['openai', 'claude']:
            raise HTTPException(status_code=400, detail="AI provider must be either 'openai' or 'claude'")
        
        email = payload.get('email')
        
        return {
            'success': True,
            'message': f'AI provider preference set to {ai_provider}',
            'ai_provider': ai_provider,
            'user_email': email
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during set AI provider")
    
@app.delete('/delete-all-sessions')
async def Delete_user_session(request: Request):
    try:
        data = await request.json()
        token = data.get('token')
        
        payload = verify_jwt_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
         
        email = payload.get('email')

        user = User()
        user.Delete_Session(email)
        
        return {
            'success': True,
            'message': f'Sucessfully Deleted Session'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during delete session")

@app.delete('/delete-session-by-date')
async def date_session_delete(request: Request):
    try:
        data = await request.json()
        token = data.get('token')
        payload = verify_jwt_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        email = payload.get('email')
        date = data.get('date')

        user = User()
        try:
            result = user.Delete_Session_By_Date(email, date)
            if not result:
                raise HTTPException(status_code=404, detail="Session not found for the given date")
        except Exception as e:
            logger.error(f"Error deleting session for {email} on {date}: {e}")
            raise HTTPException(status_code=500, detail=f"Error deleting session: {e}")

        return {
            'success': True,
            'message': 'Successfully Deleted Session'
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal error in delete-session-by-date: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during delete session")
@app.delete('/delete-platform-by-date')
async def platform_delete(request: Request):
    try:
        data = await request.json()
        token = data.get('token')
        
        payload = verify_jwt_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = payload.get('email')
        platform = data.get('platform')
        date = data.get('date')
        
        user = User()
        user.Delete_Session_By_Platform(email,date,platform)
        
        return {
            'success': True,
            'message': f'Sucessfully Deleted Session'
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during delete session")

@app.delete('/delete-date-platform-by-date')
async def date_platform_delete(request: Request):
    try:
        data = await request.json()
        token = data.get('token')
        
        payload = verify_jwt_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = payload.get('email')
        date = data.get('date')
        platform = data.get('platform')
        time = data.get('time')
        
        user = User()
        user.Delete_Session_By_Date_Platform_Time(email,date,platform,time)
        
        return {
            'success': True,
            'message': f'Sucessfully Deleted Session'
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during delete session")

@app.get('/get-platforms-by-date')
async def get_platforms_by_date(request: Request):
    try:
        data = await request.json()
        token = data.get('token')
        
        payload = verify_jwt_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = payload.get('email')
        date = data.get('date')
        
        user = User()
        platforms = user.Get_Platforms_By_Date(email,date)
        
        return {
            'success': True,
            'message': f'Sucessfully Retrieved Platforms',
            'platforms': platforms
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during get platforms")

@app.options('/get-models')
async def get_models_options():
    '''Handle preflight OPTIONS request for get-models'''
    return {"message": "OK"}
     
@app.get('/get-credits')
async def get_user_credits(request: Request):
    """Get the current credit balance for a user - only for paid users"""
    try:
        # Extract token from Authorization header or cookie
        token = None
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            # Try to get token from cookie
            token = request.cookies.get("access_token")
        
        if not token:
            raise HTTPException(status_code=401, detail="Token required")
        
        # Verify token
        user_data = verify_jwt_token(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = user_data.get('email')
        if not email:
            raise HTTPException(status_code=400, detail="Email not found in token")
        
        # Check if user is a paid user
        user = User()
        user_db_data = user.Get_User_Data(email.lower())
        
        if not user_db_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        is_paid_user = user_db_data.get('Paid_User', False)
        
        if not is_paid_user:
            # For free users, return appropriate response indicating they don't have credits
            return {
                'success': True,
                'credits': None,
                'is_paid_user': False,
                'message': 'Free users do not have credits. Please use your own API keys.',
                'email': email
            }
        
        # Get user credits only for paid users
        current_credits = credit_manager.get_user_credits(email)
        
        # Get model costs for reference
        model_costs = Credit_cost_per_minute
        
        return {
            'success': True,
            'credits': current_credits,
            'model_costs': model_costs,
            'is_paid_user': True,
            'email': email
        }
        
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Error getting credits: {e}")
        raise HTTPException(status_code=500, detail="Failed to get credits")

@app.get('/public-models')
async def get_public_models():
    """
    Return a public list of all available models (no authentication required).
    """
    try:
        # Example: Replace this with your actual model list logic if needed
        models = Credit_cost_per_minute
        return {
            'success': True,
            'message': 'Successfully retrieved public models',
            'models': models
        }
    except Exception as e:
        #logger.error(f"Error getting public models: {e}")
        raise HTTPException(status_code=500, detail="Failed to get public models")
    
@app.post('/add-credits')
async def add_user_credits(request: Request):
    """Add credits to a user's account"""
    try:
        data = await request.json()
        
        # Extract token from Authorization header or cookie
        token = None
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            token = request.cookies.get("access_token")
        
        if not token:
            raise HTTPException(status_code=401, detail="Token required")
        
        # Verify token
        user_data = verify_jwt_token(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = user_data.get('email')
        if not email:
            raise HTTPException(status_code=400, detail="Email not found in token")
        
        # Validate request data
        amount = data.get('amount')
        if not amount or not isinstance(amount, (int, float)) or amount <= 0:
            raise HTTPException(status_code=400, detail="Valid amount required (must be positive number)")
        
        reason = data.get('reason', 'manual_addition')
        
        # Add credits
        success, new_balance, message = credit_manager.add_credits(email, float(amount), reason)
        
        if success:
            # Log the transaction
            credit_manager.log_credit_transaction(email, 'add', amount, None, reason)
            
            return {
                'success': True,
                'message': message,
                'new_balance': new_balance,
                'credits_added': amount
            }
        else:
            raise HTTPException(status_code=500, detail=message)
        
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Error adding credits: {e}")
        raise HTTPException(status_code=500, detail="Failed to add credits")

# Fixed FastAPI endpoint
@app.get('/credit-history')
async def get_credit_history(request: Request):
    """Get credit transaction history for a user"""
    try:
        # Extract token from Authorization header or cookie
        token = None
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            token = request.cookies.get("access_token")
        
        if not token:
            raise HTTPException(status_code=401, detail="Token required")
        
        # Verify token
        user_data = verify_jwt_token(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = user_data.get('email')
        if not email:
            raise HTTPException(status_code=400, detail="Email not found in token")
        
        # Get user data including transaction history
        user = User()
        user_data = user.Get_User_Data(email.lower())
        
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get pagination parameters from query parameters
        pageNumber = request.query_params.get('page', '1')
        limit = request.query_params.get('limit', '5')
        
        # Convert to integers with defaults
        pageNumber = int(pageNumber)
        limit = int(limit)

        # Use the enhanced credit history method from CreditManager
        history_result = credit_manager.get_credit_history(email, pageNumber, limit)
        
        return {
            'success': True,
            'current_credits': history_result['current_credits'],
            'transactions': history_result['transactions'],
            'transaction_count': history_result['transaction_count'],
            'current_page': pageNumber,
            'limit': limit,
            'total_pages': history_result['total_pages']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Error getting credit history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get credit history")

@app.post('/check-sufficient-credits')
async def check_sufficient_credits(request: Request):
    """Check if user has sufficient credits for a specific model and duration"""
    try:
        data = await request.json()
        
        # Extract token from Authorization header or cookie
        token = None
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            token = request.cookies.get("access_token")
        
        if not token:
            raise HTTPException(status_code=401, detail="Token required")
        
        # Verify token
        user_data = verify_jwt_token(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = user_data.get('email')
        if not email:
            raise HTTPException(status_code=400, detail="Email not found in token")
        
        # Validate request data
        model = data.get('model', 'gpt-4o-mini')
        duration_minutes = data.get('duration_minutes', 1.0)
        
        if not isinstance(duration_minutes, (int, float)) or duration_minutes <= 0:
            raise HTTPException(status_code=400, detail="Valid duration_minutes required")
        
        # Check credits
        sufficient, current_balance, required_credits = credit_manager.check_sufficient_credits(
            email, model, duration_minutes
        )
        
        return {
            'success': True,
            'sufficient': sufficient,
            'current_balance': current_balance,
            'required_credits': required_credits,
            'model': model,
            'duration_minutes': duration_minutes,
            'cost_per_minute': credit_manager.calculate_total_cost_per_minute(model)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Error checking credits: {e}")
        raise HTTPException(status_code=500, detail="Failed to check credits")

def check_user_has_api_keys(email: str) -> bool:
    """
    Check if user has at least one valid API key configured
    
    Args:
        email: User email to check API keys for
        
    Returns:
        bool: True if user has at least one non-empty API key, False otherwise
    """
    try:
        api_keys = get_User_API_Key(email)
        if not api_keys or not isinstance(api_keys, dict):
            return False
        
        # Check if any of the API keys are present and not empty
        openai_key = api_keys.get('OpenAI', '').strip()
        deepgram_key = api_keys.get('Deepgram', '').strip()
        claude_key = api_keys.get('Claude', '').strip()
        gemini_key = api_keys.get('Gemini', '').strip()
        
        return bool(openai_key or deepgram_key or claude_key or gemini_key)
    except Exception as e:
        #logger.error(f"Error checking API keys for {email}: {str(e)}")
        return False

@app.post('/check-transcription-eligibility')
async def check_transcription_eligibility(request: Request):
    """
    Check if user is eligible to start transcription based on:
    1. Subscription status (paid vs free user)
    2. Credit availability for paid users
    3. API key requirements for free users
    
    Returns comprehensive status with appropriate messages
    """
    try:
        # Extract token from Authorization header or cookie
        token = None
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            token = request.cookies.get("access_token")
        
        if not token:
            raise HTTPException(status_code=401, detail="Token required")
        
        # Verify token
        user_data = verify_jwt_token(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = user_data.get('email')
        if not email:
            raise HTTPException(status_code=400, detail="Email not found in token")
        
        # Get request data
        data = await request.json()
        model = data.get('model', 'gpt-4o-mini')
        estimated_duration = data.get('estimated_duration_minutes', 1.0)
        
        # Get user information
        user = User()
        user_db_data = user.Get_User_Data(email.lower())
        
        if not user_db_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Check if user is a paid user
        is_paid_user = user_db_data.get('Paid_User', False)
        
        response_data = {
            'success': True,
            'is_paid_user': is_paid_user,
            'eligible_for_transcription': False,
            'message': '',
            'current_credits': 0,
            'required_credits': 0,
            'has_api_keys': False
        }
        
        if is_paid_user:
            # For paid users: check credits only, don't require API keys
            current_balance = credit_manager.get_user_credits(email)
            sufficient, _, required_credits = credit_manager.check_sufficient_credits(
                email, model, estimated_duration
            )
            
            response_data.update({
                'current_credits': current_balance,
                'required_credits': required_credits,
                'has_api_keys': True  # Paid users use system API keys
            })
            
            if sufficient:
                response_data.update({
                    'eligible_for_transcription': True,
                    'message': 'Transcription will begin now. Your credits will be deducted accordingly.'
                })
            else:
                response_data.update({
                    'eligible_for_transcription': False,
                    'message': 'Your credits are below the required threshold for transcription. Please top up your credits to continue.'
                })
        else:
            # For free users: check if they have API keys configured
            has_api_keys = check_user_has_api_keys(email)
            response_data.update({
                'has_api_keys': has_api_keys,
                'current_credits': 'N/A (Free User)',
                'required_credits': 'N/A (Free User)'
            })
            
            if has_api_keys:
                response_data.update({
                    'eligible_for_transcription': True,
                    'message': 'Transcription will begin using your configured API keys.'
                })
            else:
                response_data.update({
                    'eligible_for_transcription': False,
                    'message': 'Please provide your API keys in the settings to proceed with the transcription.'
                })
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Error checking transcription eligibility: {e}")
        raise HTTPException(status_code=500, detail="Failed to check transcription eligibility")

# Stripe webhook endpoints
@app.post('/stripe/webhook')
async def stripe_webhook(request: Request):
    """
    Handle Stripe webhook events for payment processing.
    This endpoint processes payment confirmations and allocates credits to users.
    """
    try:
        # Get the raw body and signature
        payload = await request.body()
        signature = request.headers.get('stripe-signature')
        
        if not signature:
            raise HTTPException(status_code=400, detail="Missing stripe-signature header")
        
        # Get Stripe manager and verify webhook
        stripe_manager = get_stripe_manager()
        event = stripe_manager.verify_webhook_signature(payload, signature)
        
        #logger.info(f"Received Stripe webhook event: {event['type']}")
        
        # Handle different event types
        if event['type'] in ['payment_intent.succeeded', 'checkout.session.completed']:
            result = stripe_manager.handle_payment_success(event)
            #logger.info(f"Payment success result: {result}")
            return {"status": "success", "message": "Payment processed successfully"}
            
        elif event['type'] == 'payment_intent.payment_failed':
            result = stripe_manager.handle_payment_failed(event)
            #logger.info(f"Payment failure result: {result}")
            return {"status": "success", "message": "Payment failure logged"}
            
        else:
            #logger.info(f"Unhandled event type: {event['type']}")
            return {"status": "success", "message": "Event received but not processed"}
            
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Error processing Stripe webhook: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process webhook")

@app.post('/stripe/create-payment-intent')
async def create_payment_intent(request: Request):
    """
    Create a Stripe payment intent for custom payment flows.
    """
    try:
        data = await request.json()
        
        # Extract token to get user info
        token = None
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            token = request.cookies.get("access_token")
        
        if not token:
            raise HTTPException(status_code=401, detail="Token required")
        
        # Verify token
        user_data = verify_jwt_token(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = user_data.get('email')
        amount = data.get('amount')  # Amount in cents
        currency = data.get('currency', 'usd')
        
        if not amount or not isinstance(amount, int) or amount <= 0:
            raise HTTPException(status_code=400, detail="Valid amount in cents required")
        
        # Create payment intent
        stripe_manager = get_stripe_manager()
        result = stripe_manager.create_payment_intent(
            amount=amount,
            currency=currency,
            metadata={'customer_email': email}
        )
        
        if result['status'] == 'success':
            return {
                'success': True,
                'client_secret': result['client_secret']
            }
        else:
            raise HTTPException(status_code=500, detail=result['message'])
            
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Error creating payment intent: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create payment intent")

@app.post('/stripe/purchase-credits')
async def purchase_credits(request: Request):
    """
    Create a payment intent for purchasing specific credit amounts from the frontend pricing page.
    """
    try:
        data = await request.json()
        
        # Extract token to get user info
        token = None
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            token = request.cookies.get("access_token")
        
        if not token:
            raise HTTPException(status_code=401, detail="Token required")
        
        # Verify token
        user_data = verify_jwt_token(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = user_data.get('email')
        credits = data.get('credits')
        price = data.get('price')
        
        if not credits or not isinstance(credits, int) or credits <= 0:
            raise HTTPException(status_code=400, detail="Valid credits amount required")
        
        if not price or not isinstance(price, (int, float)) or price <= 0:
            raise HTTPException(status_code=400, detail="Valid price required")
        
        # Create payment intent for credit purchase
        stripe_manager = get_stripe_manager()
        result = stripe_manager.create_payment_intent_for_credits(email, credits, price)
        
        if result['status'] == 'success':
            return {
                'success': True,
                'client_secret': result['client_secret'],
                'payment_intent_id': result['payment_intent_id']
            }
        else:
            raise HTTPException(status_code=500, detail=result['message'])
            
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Error creating credit purchase payment intent: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create credit purchase payment intent")

@app.post('/stripe/create-checkout-session')
async def create_checkout_session(request: Request):
    """
    Create a Stripe checkout session for package purchases.
    """
    try:
        data = await request.json()
        
        # Extract token to get user info
        token = None
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            token = request.cookies.get("access_token")
        
        if not token:
            raise HTTPException(status_code=401, detail="Token required")
        
        # Verify token
        user_data = verify_jwt_token(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = user_data.get('email')
        package_name = data.get('package_name')
        return_url = data.get('return_url', 'http://localhost:5173/payment-details')
        cancel_url = data.get('cancel_url', 'http://localhost:5173/pricing')
        
        if not package_name:
            raise HTTPException(status_code=400, detail="Package name required")
        
        # Create checkout session
        stripe_manager = get_stripe_manager()
        result = stripe_manager.create_checkout_session_for_package(
            email, package_name, return_url, cancel_url
        )
        
        #logger.info(f"Stripe result: {result}")

        
        if result['status'] == 'success':
            return {
                'success': True,
                'checkout_url': result['checkout_url'],
                'session_id': result['session_id']
            }
        else:
            raise HTTPException(status_code=500, detail=result['message'])
            
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Error creating checkout session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create checkout session")

@app.get('/stripe/payment-status/{payment_intent_id}')
async def get_payment_status(payment_intent_id: str, request: Request):
    """
    Get the status of a specific payment intent.
    """
    try:
        # Extract token for authentication
        token = None
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            token = request.cookies.get("access_token")
        
        if not token:
            raise HTTPException(status_code=401, detail="Token required")
        
        # Verify token
        user_data = verify_jwt_token(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Get payment status
        stripe_manager = get_stripe_manager()
        result = stripe_manager.retrieve_payment_status(payment_intent_id)
        
        if result['status'] == 'success':
            return {
                'success': True,
                'payment_status': result['payment_status'],
                'amount': result['amount'],
                'currency': result['currency'],
                'metadata': result.get('metadata', {})
            }
        else:
            raise HTTPException(status_code=500, detail=result['message'])
            
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Error getting payment status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get payment status")

@app.get('/stripe/packages')
async def get_credit_packages():
    """
    Get available credit packages for the frontend pricing page.
    """
    try:
        # Format packages for frontend consumption
        formatted_packages = []
        
        for package_name, package_data in Packages.items():
            formatted_packages.append({
                'name': package_name,
                'credits': package_data['Credits'],
                'price': package_data['Price'],
                'description': package_data['Description'],
                'features': package_data.get('Features', []),
                'popular': package_name.lower() == 'pro',  # Mark Pro as popular
                'discount': '20% off' if package_name.lower() == 'pro' else ('30% off' if package_name.lower() == 'enterprise' else None),
                'stripe_link': package_data.get('Link', package_data.get('link'))
            })
        
        return {
            'success': True,
            'packages': formatted_packages
        }
        
    except Exception as e:
        #logger.error(f"Error getting credit packages: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get credit packages")

@app.get('/get-transaction-details/{identifier}')
async def get_transaction_details(identifier: str, request: Request):
    """
    Get transaction details for a specific Stripe identifier (Customer ID, session ID, or payment intent ID).
    Now retrieves transaction details directly from Stripe and saves to database if not already present.
    Used by the payment-details page to show invoice information.
    """
    try:
        #logger.info(f"Received request for transaction details with identifier: {identifier}")
        
        # Extract and verify token
        token = None
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            token = request.cookies.get("access_token")
        
        if not token:
            #logger.error("No token provided in request")
            raise HTTPException(status_code=401, detail="Token required")
        
        user_data = verify_jwt_token(token)
        if not user_data:
            #logger.error("Invalid token provided")
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = user_data.get('email').lower()
        #logger.info(f"Looking for transaction for user: {email}")
        
        # Get user from database
        user = Find_User_DB(email)
        if not user:
            #logger.error(f"User not found in database: {email}")
            raise HTTPException(status_code=404, detail="User not found")
        
        # Check if this identifier has already been processed to prevent duplicate processing
        existing_transactions = user.get('CreditTransactions', [])
        
        # Ensure existing_transactions is a list (handle corrupted data)
        if not isinstance(existing_transactions, list):
            #logger.warning(f"CreditTransactions field is corrupted for user {email}, treating as empty")
            existing_transactions = []
        
        # Check if we already have a transaction for this identifier
        for transaction in existing_transactions:
            if (transaction.get('stripe_session_id') == identifier or 
                transaction.get('stripe_payment_intent_id') == identifier or
                transaction.get('stripe_customer_id') == identifier):
                #logger.info(f"Transaction already processed for identifier {identifier}")
                
                # Return the existing transaction details without processing again
                return {
                    "success": True,
                    "already_processed": True,
                    "transaction": transaction,
                    "invoice_details": transaction.get('invoice_data', {}),
                    "search_method": "already_processed",
                    "message": f"Transaction has already been processed for identifier {identifier}",
                    "user_info": {
                        "name": user.get('Name', ''),
                        "email": user.get('Email', ''),
                        "current_credits": user.get('Credits', 0)
                    }
                }
        
        # Import stripe manager
        from Stripe import get_stripe_manager
        stripe_manager = get_stripe_manager()
        
        # Step 1: Get transaction details directly from Stripe
        #logger.info(f"Retrieving transaction details from Stripe for identifier: {identifier}")
        stripe_result = stripe_manager.get_transaction_details_from_stripe(identifier)
        
        if stripe_result["status"] != "success":
            #logger.error(f"Failed to get transaction from Stripe: {stripe_result.get('message')}")
            
            # Fallback: Check if transaction exists in local database
            #logger.info("Falling back to local database search")
            transactions = user.get('CreditTransactions', [])
            
            # Ensure transactions is a list (handle corrupted data)
            if not isinstance(transactions, list):
                #logger.warning(f"CreditTransactions field is corrupted for user {email} during fallback search")
                transactions = []
            
            for trans in transactions:
                if (trans.get('stripe_session_id') == identifier or 
                    trans.get('stripe_payment_intent_id') == identifier or
                    trans.get('stripe_customer_id') == identifier):
                    #logger.info(f"Found transaction in local database")
                    return {
                        "success": True,
                        "transaction": trans,
                        "invoice_details": trans.get('invoice_data', {}),
                        "search_method": "local_database",
                        "user_info": {
                            "name": user.get('Name', ''),
                            "email": user.get('Email', ''),
                            "current_credits": user.get('Credits', 0)
                        }
                    }
            
            # If neither Stripe nor local database has the transaction
            #logger.warning(f"Transaction not found in Stripe or local database for identifier: {identifier}")
            return {
                "success": True,
                "processing": True,
                "message": "Payment is being processed. Please wait a moment and refresh if needed.",
                "user_info": {
                    "name": user.get('Name', ''),
                    "email": user.get('Email', ''),
                    "current_credits": user.get('Credits', 0)
                }
            }
        
        transaction_data = stripe_result["transaction"]
        
        # Step 2: Save transaction to database if not already present
        saved = stripe_manager.save_transaction_to_database(email, transaction_data)


        # Step 3: Update user credits if transaction is new (not already processed)
        user = Find_User_DB(email)  # Refresh user data after potential save
        existing_transactions = user.get('CreditTransactions', [])
        
        # Ensure existing_transactions is a list (handle corrupted data)
        if not isinstance(existing_transactions, list):
            #logger.warning(f"CreditTransactions field is corrupted for user {email}, treating as empty")
            existing_transactions = []
        
        credits_already_added = False
        
        # Only check for duplicates if we have transactions to check
        if len(existing_transactions) > 1:
            try:
                # Check all transactions except the potentially just-added one
                for existing_trans in existing_transactions[:-1]:
                    if (transaction_data.get('stripe_session_id') and 
                        existing_trans.get('stripe_session_id') == transaction_data.get('stripe_session_id')) or \
                       (transaction_data.get('stripe_payment_intent_id') and 
                        existing_trans.get('stripe_payment_intent_id') == transaction_data.get('stripe_payment_intent_id')):
                        credits_already_added = True
                        break
            except Exception as slice_error:
                #logger.warning(f"Error checking existing transactions for duplicates: {str(slice_error)}")
                # Fall back to checking all transactions
                for existing_trans in existing_transactions:
                    if (transaction_data.get('stripe_session_id') and 
                        existing_trans.get('stripe_session_id') == transaction_data.get('stripe_session_id')) or \
                       (transaction_data.get('stripe_payment_intent_id') and 
                        existing_trans.get('stripe_payment_intent_id') == transaction_data.get('stripe_payment_intent_id')):
                        credits_already_added = True
                        break
        
        # Add credits if this is a new transaction
        if not credits_already_added and transaction_data.get('amount', 0) > 0:
            from payment import CreditManager
            credit_manager = CreditManager()
            
            success, new_balance, message = credit_manager.add_credits(
                email,
                transaction_data['amount'],
                transaction_data.get('reason', 'Stripe payment'),
                auto_log=False  # Don't log again since we already saved the transaction
            )
            
            if success:
                #logger.info(f"Added {transaction_data['amount']} credits to user {email}")
                # Update user data for response
                user = Find_User_DB(email)
            else:
                #logger.warning(f"Failed to add credits: {message}")
                pass
        else:
            pass
            #logger.info(f"Credits already added for this transaction or amount is 0 for user {email}")
        
        # Return transaction details
        return {
            "success": True,
            "transaction": transaction_data,
            "invoice_details": transaction_data.get('invoice_data', {}),
            "search_method": "stripe_direct",
            "user_info": {
                "name": user.get('Name', ''),
                "email": user.get('Email', ''),
                "current_credits": user.get('Credits', 0)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Error getting transaction details: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get transaction details")

@app.post('/test-payment-webhook/{session_id}')
async def test_payment_webhook(session_id: str, request: Request):
    """
    Test endpoint to simulate a successful payment webhook for debugging.
    This helps test the payment flow without going through Stripe.
    """
    try:
        #logger.info(f"Test webhook triggered for session_id: {session_id}")
        
        # Extract and verify token
        token = None
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            token = request.cookies.get("access_token")
        
        if not token:
            raise HTTPException(status_code=401, detail="Token required")
        
        user_data = verify_jwt_token(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = user_data.get('email').lower()
        #logger.info(f"Testing payment for user: {email}")
        
        # Check if this test payment has already been processed to prevent duplicates
        from Autentication import collection
        existing_user = collection.find_one({"Email": email})
        
        if existing_user:
            credit_transactions = existing_user.get('CreditTransactions', [])
            
            # Check if we already have a test transaction for this session ID
            for transaction in credit_transactions:
                if (transaction.get('stripe_session_id') == session_id and 
                    transaction.get('package_name') == 'Pro'):
                    #logger.info(f"Test payment already processed for session {session_id}")
                    
                    return {
                        "success": True,
                        "already_processed": True,
                        "message": f"Test payment has already been processed for session {session_id}",
                        "new_balance": existing_user.get('Credits', 0)
                    }
        
        # Ensure customer ID is saved for test payment
        customer_id_from_stripe = await ensure_customer_id_saved(email, session_id)
        
        # Create a test transaction entry
        from datetime import datetime
        test_transaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "purchase",
            "amount": 5000.0,  # 5000 credits
            "stripe_session_id": session_id,
            "package_name": "Pro",
            "package_price": 40.0,
            "payment_currency": "USD",
            "payment_method": "stripe",
            "payment_status": "completed",
            "invoice_data": {
                "transaction_id": session_id,
                "customer_email": email,
                "amount_paid": 40.0,
                "credits_purchased": 5000,
                "package_description": "Most popular choice",
                "payment_date": datetime.utcnow().isoformat()
            }
        }
        
        # Add credits to user
        success, new_balance, message = credit_manager.add_credits(
            email,
            5000,
            "Test payment - Pro package",
            auto_log=False
        )
        
        if success:
            # Add transaction to user's record
            from Autentication import collection
            collection.update_one(
                {"Email": email},
                {"$push": {"CreditTransactions": test_transaction}},
                upsert=False
            )
            
            #logger.info(f"Test payment successful for {email}")
            return {
                "success": True,
                "message": f"Test payment processed successfully. Added 5000 credits to {email}",
                "new_balance": new_balance
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to add credits: {message}")
            
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Error in test payment webhook: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process test payment")

@app.post('/manual-payment-completion')
async def manual_payment_completion(request: Request):
    """
    Manually trigger payment completion for a session ID.
    This is a temporary solution for testing when webhooks aren't working in development.
    """
    try:
        data = await request.json()
        session_id = data.get('session_id')
        
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id required")
        
        # Extract and verify token
        token = None
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            token = request.cookies.get("access_token")
        
        if not token:
            raise HTTPException(status_code=401, detail="Token required")
        
        user_data = verify_jwt_token(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = user_data.get('email').lower()
        #logger.info(f"Manual payment completion for session_id: {session_id}, user: {email}")
        
        # Check if this payment has already been processed to prevent duplicates
        from Autentication import collection
        existing_user = collection.find_one({"Email": email})
        
        if existing_user:
            credit_transactions = existing_user.get('CreditTransactions', [])
            
            # Check if we already have a transaction for this session ID
            for transaction in credit_transactions:
                if transaction.get('stripe_session_id') == session_id:
                    #logger.info(f"Payment already processed for session {session_id}")
                    
                    # Return the existing payment details without processing again
                    return {
                        "success": True,
                        "already_processed": True,
                        "message": f"Payment has already been processed for session {session_id}",
                        "new_balance": existing_user.get('Credits', 0),
                        "transaction": transaction
                    }
        
        # Ensure customer ID is saved before processing
        customer_id_from_stripe = await ensure_customer_id_saved(email, session_id)
        
        # Get Stripe session details to extract payment information
        stripe_manager = get_stripe_manager()
        
        try:
            import stripe
            session = stripe.checkout.Session.retrieve(session_id)
            
            if session.payment_status != 'paid':
                raise HTTPException(status_code=400, detail="Payment not completed in Stripe")
            
            # Extract metadata from the session
            metadata = session.metadata or {}
            customer_email = metadata.get('customer_email', email)
            package_name = metadata.get('package_name', 'Unknown')
            credits_str = metadata.get('credits', '0')
            
            try:
                credits = int(credits_str)
            except:
                credits = 0
            
            # Get package price from amount_total (convert from cents to dollars)
            package_price = session.amount_total / 100.0 if session.amount_total else 0.0
            
            #logger.info(f"Processing payment: {credits} credits, ${package_price}, package: {package_name}")
            
            # Create transaction data
            from datetime import datetime
            transaction_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": "purchase",
                "amount": float(credits),
                "stripe_session_id": session_id,
                "stripe_payment_intent_id": session.payment_intent,
                "package_name": package_name,
                "package_price": package_price,
                "payment_currency": session.currency.upper() if session.currency else "USD",
                "payment_method": "stripe",
                "payment_status": "completed",
                "invoice_data": {
                    "transaction_id": session_id,
                    "customer_email": customer_email,
                    "amount_paid": package_price,
                    "credits_purchased": credits,
                    "package_description": f"{package_name} Package - {credits} Credits",
                    "payment_date": datetime.utcnow().isoformat()
                }
            }
            
            # Add credits to user account
            success, new_balance, message = credit_manager.add_credits(
                customer_email,
                credits,
                f"Manual payment completion - {package_name} package",
                auto_log=False
            )
            
            if not success:
                raise HTTPException(status_code=500, detail=f"Failed to add credits: {message}")
            
            # Add transaction to user's record
            from Autentication import collection
            result = collection.update_one(
                {"Email": customer_email},
                {"$push": {"CreditTransactions": transaction_data}},
                upsert=False
            )
            
            return {
                "success": True,
                "message": f"Payment processed successfully. Added {credits} credits.",
                "new_balance": new_balance,
                "transaction": transaction_data
            }
            
        except stripe.error.StripeError as e:
            #logger.error(f"Stripe error retrieving session: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid session ID or Stripe error: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Error in manual payment completion: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process manual payment completion")

@app.get('/get-customer-transactions')
async def get_customer_transactions(request: Request):
    """
    Get all transaction details for the authenticated user using their Stripe Customer ID.
    This is the preferred method for retrieving user transactions.
    """
    try:
        # Extract and verify token
        token = None
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            token = request.cookies.get("access_token")
        
        if not token:
            raise HTTPException(status_code=401, detail="Token required")
        
        user_data = verify_jwt_token(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = user_data.get('email').lower()
        #logger.info(f"Getting customer transactions for user: {email}")
        
        # Get user from database
        user = Find_User_DB(email)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get user's Stripe Customer ID
        stripe_customer_id = user.get('Stripe_Customer_ID')
        
        if not stripe_customer_id:
            return {
                "success": True,
                "message": "No Stripe Customer ID found for user",
                "customer_id": None,
                "transactions": [],
                "user_info": {
                    "name": user.get('Name', ''),
                    "email": user.get('Email', ''),
                    "current_credits": user.get('Credits', 0)
                }
            }
        
        # Get all transactions for this customer
        all_transactions = user.get('CreditTransactions', [])
        customer_transactions = [
            t for t in all_transactions 
            if t.get('stripe_customer_id') == stripe_customer_id
        ]
        
        # Sort by timestamp (most recent first)
        customer_transactions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        #logger.info(f"Found {len(customer_transactions)} transactions for customer {stripe_customer_id}")
        
        return {
            "success": True,
            "customer_id": stripe_customer_id,
            "transactions": customer_transactions,
            "transaction_count": len(customer_transactions),
            "user_info": {
                "name": user.get('Name', ''),
                "email": user.get('Email', ''),
                "current_credits": user.get('Credits', 0)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Error getting customer transactions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get customer transactions")

@app.get('/get-stripe-customer-id')
async def get_stripe_customer_id(request: Request):
    """
    Get the Stripe Customer ID for the authenticated user.
    """
    try:
        # Extract and verify token
        token = None
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            token = request.cookies.get("access_token")
        
        if not token:
            raise HTTPException(status_code=401, detail="Token required")
        
        user_data = verify_jwt_token(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = user_data.get('email').lower()
        
        # Get user from database
        user = Find_User_DB(email)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        stripe_customer_id = user.get('Stripe_Customer_ID')
        
        return {
            "success": True,
            "customer_id": stripe_customer_id,
            "has_customer_id": stripe_customer_id is not None,
            "email": email
        }
        
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Error getting Stripe customer ID: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get Stripe customer ID")

async def ensure_customer_id_saved(email: str, identifier: str) -> str:
    """
    Ensure that the user's Stripe Customer ID is saved in the database.
    If not present, retrieve it from Stripe and save it.
    
    Args:
        email (str): User email
        identifier (str): Stripe identifier (session_id, payment_intent_id, etc.)
        
    Returns:
        str: Stripe Customer ID if found, None otherwise
    """
    try:
        from User import User
        user_obj = User()
        
        # Check if user already has Customer ID saved
        existing_customer_id = user_obj.Get_Stripe_Customer_ID(email)
        if existing_customer_id:
            #logger.info(f"User {email} already has Customer ID: {existing_customer_id}")
            return existing_customer_id
        
        #logger.info(f"No Customer ID found for {email}, attempting to retrieve from Stripe")
        
        # Try to get customer ID from Stripe using the identifier
        stripe_manager = get_stripe_manager()
        customer_id = None
        
        # If identifier looks like a session ID, retrieve session details
        if identifier.startswith('cs_'):
            try:
                import stripe
                session = stripe.checkout.Session.retrieve(identifier)
                customer_id = session.get('customer')
                #logger.info(f"Retrieved customer ID {customer_id} from session {identifier}")
            except Exception as e:
                pass
        
        # If identifier looks like a payment intent ID, retrieve payment intent details
        elif identifier.startswith('pi_'):
            try:
                import stripe
                payment_intent = stripe.PaymentIntent.retrieve(identifier)
                customer_id = payment_intent.get('customer')
                #logger.info(f"Retrieved customer ID {customer_id} from payment intent {identifier}")
            except Exception as e:
                pass
        
        # If we found a customer ID, save it to the database
        if customer_id:
            saved = user_obj.Update_Stripe_Customer_ID(email, customer_id)
            if saved:
                return customer_id
            else:
                pass
        else:
            customer_result = stripe_manager.create_or_get_customer(email)
            if customer_result['status'] == 'success':
                customer_id = customer_result['customer_id']
                saved = user_obj.Update_Stripe_Customer_ID(email, customer_id)
                if saved:
                    return customer_id
                else:
                    pass
        
        return None
        
    except Exception as e:
        #logger.error(f"Error ensuring customer ID for {email}: {str(e)}")
        return None

@app.post('/process-stripe-return')
async def process_stripe_return(request: Request):
    """
    Process user return from Stripe payment.
    Ensures customer ID is retrieved and saved to database.
    Should be called when user returns from Stripe checkout.
    """
    try:
        data = await request.json()
        
        # Extract and verify token
        token = None
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            token = request.cookies.get("access_token")
        
        if not token:
            raise HTTPException(status_code=401, detail="Token required")
        
        user_data = verify_jwt_token(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        email = user_data.get('email').lower()
        session_id = data.get('session_id')
        
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        
        #logger.info(f"Processing Stripe return for user: {email}, session: {session_id}")
        
        # Ensure customer ID is saved
        customer_id = await ensure_customer_id_saved(email, session_id)
        
        if customer_id:
            #logger.info(f"Successfully processed Stripe return. Customer ID: {customer_id}")
            return {
                "success": True,
                "message": "Stripe return processed successfully",
                "customer_id": customer_id,
                "session_id": session_id
            }
        else:
            #logger.warning(f"Could not retrieve/save customer ID for session: {session_id}")
            return {
                "success": True,
                "message": "Stripe return processed, but customer ID not available",
                "customer_id": None,
                "session_id": session_id
            }
        
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Error processing Stripe return: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process Stripe return")

@app.post('/admin/fix-credit-transactions')
async def fix_credit_transactions(request: Request):
    """
    Admin endpoint to fix corrupted CreditTransactions fields in the database.
    """
    try:
        # Extract and verify token
        token = None
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            token = request.cookies.get("access_token")
        
        if not token:
            #logger.error("No token provided in request")
            raise HTTPException(status_code=401, detail="Token required")
        
        user_data = verify_jwt_token(token)
        if not user_data:
            #logger.error("Invalid token provided")
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Import stripe manager
        from Stripe import get_stripe_manager
        stripe_manager = get_stripe_manager()
        
        # Run database repair
        #logger.info("Starting database repair for corrupted CreditTransactions")
        results = stripe_manager.fix_all_corrupted_credit_transactions()
        
        return {
            "success": True,
            "message": "Database repair completed",
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Error fixing credit transactions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fix credit transactions")

@app.post('/admin/fix-user-credit-transactions/{email}')
async def fix_user_credit_transactions(email: str, request: Request):
    """
    Admin endpoint to fix corrupted CreditTransactions field for a specific user.
    """
    try:
        # Extract and verify token
        token = None
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            token = request.cookies.get("access_token")
        
        if not token:
            #logger.error("No token provided in request")
            raise HTTPException(status_code=401, detail="Token required")
        
        user_data = verify_jwt_token(token)
        if not user_data:
            #logger.error("Invalid token provided")
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Import stripe manager
        from Stripe import get_stripe_manager
        stripe_manager = get_stripe_manager()
        
        # Fix specific user
        #logger.info(f"Fixing CreditTransactions for user: {email}")
        success = stripe_manager.fix_user_credit_transactions(email)
        
        if success:
            return {
                "success": True,
                "message": f"Successfully fixed CreditTransactions for user {email}"
            }
        else:
            return {
                "success": False,
                "message": f"Failed to fix CreditTransactions for user {email}"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        #logger.error(f"Error fixing credit transactions for user {email}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fix user credit transactions")

@app.websocket('/listen')
async def websocket_listen(websocket: WebSocket):
    assistant = None
    try:
        # Accept WebSocket connection with keep-alive settings
        await websocket.accept()
        try:
            # Set a shorter timeout for auth message to speed up connection
            auth_message = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
            if not auth_message or auth_message.get('type') != 'authorization':
                await safe_websocket_send(websocket, {
                    'type': 'error',
                    'message': 'Authorization message required as first message'
                })
                await safe_websocket_close(websocket, code=4001, reason="Missing authorization")
                return
        except asyncio.TimeoutError:
            await safe_websocket_close(websocket, code=4001, reason="Authorization timeout")
            return
        except json.JSONDecodeError:
            await safe_websocket_send(websocket, {
                'type': 'error',
                'message': 'Invalid JSON in authorization message'
            })
            await safe_websocket_close(websocket, code=4001, reason="Invalid JSON")
            return
        except Exception as e:
            await safe_websocket_close(websocket, code=4001, reason="Authorization receive error")
            return
        
        token = auth_message.get('token')
        if not token:
            await safe_websocket_send(websocket, {
                'type': 'error',
                'message': 'JWT token is required for authorization'
            })
            await safe_websocket_close(websocket, code=4001, reason="Missing JWT token")
            return
            
        try:
            #logger.info(f"Token received: {token}")
            if token.startswith('Bearer '):
                token = token[7:]
            
            payload = verify_jwt_token(token)
            #logger.info(f"Payload received: {payload}")
            user_email = payload.get('email')
            
            #logger.info(f"User email received: {user_email}")
            if not user_email:
                await safe_websocket_send(websocket, {
                    'type': 'error',
                    'message': 'Invalid token: email not found'
                })
                await safe_websocket_close(websocket, code=4001, reason="Invalid token")
                return
                
        except HTTPException as e:
            await safe_websocket_send(websocket, {
                'type': 'error',
                'message': f'Token verification failed: {e.detail}'
            })
            await safe_websocket_close(websocket, code=4001, reason="Token verification failed")
            return
        except Exception as e:
            #logger.error(f"Unexpected error during token verification: {str(e)}")
            await safe_websocket_send(websocket, {
                'type': 'error',
                'message': 'Internal error during token verification'
            })
            await safe_websocket_close(websocket, code=4001, reason="Token verification error")
            return

        # Extract configuration parameters
        try:
            
            dg_api_key = os.getenv("DEEPGRAM_API_KEY")

            # getting this information from database
            user = User()
            user_data = user.Get_User_Data(user_email)
            
            # Check if user is a paid user
            is_paid_user = user_data.get('Paid_User', False)

            # Get selected model from auth_message or fallback to query params
            query_params = websocket.query_params
            selected_model = auth_message.get('selected_model')
            if not selected_model:
                selected_model = query_params.get('modelname', 'gpt-4o-mini')
            
            # For paid users, verify credits before starting transcription
            if is_paid_user:
                sufficient, current_balance, required_credits = credit_manager.check_sufficient_credits(
                    user_email, selected_model, 1.0
                )
                
                if not sufficient:
                    await safe_websocket_send(websocket, {
                        'type': 'error',
                        'message': f'Your credits are below the required threshold for transcription. Please top up your credits to continue. Required: {required_credits:.2f}, Available: {current_balance:.2f}'
                    })
                    await safe_websocket_close(websocket, code=4006, reason="Insufficient credits")
                    return
                
                await safe_websocket_send(websocket, {
                    'type': 'credit_status',
                    'current_balance': current_balance,
                    'required_per_minute': required_credits,
                    'model': selected_model,
                    'message': 'Transcription will begin now. Your credits will be deducted accordingly.'
                })
            else:
                # For free users, verify API keys are configured
                has_api_keys = check_user_has_api_keys(user_email)
                if not has_api_keys:
                    await safe_websocket_send(websocket, {
                        'type': 'error',
                        'message': 'Please provide your API keys in the settings to proceed with the transcription.'
                    })
                    await safe_websocket_close(websocket, code=4007, reason="Missing API keys")
                    return
                

            if not is_paid_user:
                # Use user's own API keys
                user_api_keys = user_data.get('API_Key', {})
                dg_api_key = user_api_keys.get('Deepgram')
                openai_api_key = user_api_keys.get('OpenAI')
                claude_api_key = user_api_keys.get('Claude')
                gemini_api_key = user_api_keys.get('Gemini')
            else:
                # Use system API keys for paid users
                dg_api_key = os.getenv("DEEPGRAM_API_KEY")
                openai_api_key = os.getenv("OPENAI_API_KEY")
                claude_api_key = os.getenv("CLAUDE_API_KEY")
                gemini_api_key = os.getenv("GEMINI_API_KEY")

            ai_mode = user_data.get('Summary_and_Keywords')
            
            CHAT_GPT_MODEL_BEING_USED = selected_model
            # ai_provider = auth_message.get('ai_provider', 'openai').lower()

            # check if the model has 'gpt' in it then provider is openai
            if 'gpt' in selected_model:
                ai_provider = 'openai'
            elif 'claude' in selected_model:
                ai_provider = 'claude'
            elif 'gemini' in selected_model:
                ai_provider = 'gemini'
            else:
                ai_provider = 'openai'
            source_language = query_params.get('source', 'English') 
            target_language = query_params.get('target', 'French')
            mode = query_params.get('mode', 'speed')
            translation_provider = query_params.get('provider', 'openai').lower()
            web_page_name = auth_message.get('web_page_name', 'Unknown')
            
        except Exception as e:
            #logger.error(f"Error extracting configuration parameters: {str(e)}")
            await safe_websocket_send(websocket, {
                'type': 'error',
                'message': 'Invalid configuration parameters'
            })
            await safe_websocket_close(websocket, code=4002, reason="Configuration error")
            return
        
        # Validate translation provider
        if translation_provider not in ['openai', 'claude', 'gemini', 'deepgram']:
            await safe_websocket_send(websocket, {
                'type': 'error',
                'message': 'Translation provider must be either "openai", "claude", "gemini", or "deepgram"'
            })
            await safe_websocket_close(websocket, code=4003, reason="Invalid translation provider")
            return
        
        # Validate Deepgram API key
        if not dg_api_key or dg_api_key.strip() == "" or dg_api_key == "your_deepgram_api_key_here":
            #logger.warning("DEEPGRAM_API_KEY is not set or invalid. Audio transcription will be disabled.")
            await safe_websocket_send(websocket, {
                'type': 'warning',
                'message': 'Deepgram API key not configured. Transcription may be limited.'
            })
            dg_api_key = None
        
        # Validate API keys based on provider
        if ai_provider == 'openai':
            if not openai_api_key or openai_api_key.strip() == "":
                await safe_websocket_send(websocket, {
                    'type': 'error',
                    'message': 'OpenAI API key is required when using OpenAI as AI provider'
                })
                await safe_websocket_close(websocket, code=4003, reason="Missing OpenAI API key")
                return
        elif ai_provider == 'claude':
            if not claude_api_key or claude_api_key.strip() == "":
                await safe_websocket_send(websocket, {
                    'type': 'error',
                    'message': 'Claude API key is required when using Claude as AI provider'
                })
                await safe_websocket_close(websocket, code=4003, reason="Missing Claude API key")
                return
        elif ai_provider == 'gemini':
            if not gemini_api_key or gemini_api_key.strip() == "":
                await safe_websocket_send(websocket, {
                    'type': 'error',
                    'message': 'Gemini API key is required when using Gemini as AI provider'
                })
                await safe_websocket_close(websocket, code=4003, reason="Missing Gemini API key")
                return
            
        # Validate translation provider API keys
        if translation_provider == 'openai' and (not openai_api_key or openai_api_key.strip() == ""):
            await safe_websocket_send(websocket, {
                'type': 'error',
                'message': 'OpenAI API key is required for OpenAI translation provider'
            })
            await safe_websocket_close(websocket, code=4003, reason="Missing OpenAI API key for translation")
            return
        elif translation_provider == 'claude' and (not claude_api_key or claude_api_key.strip() == ""):
            await safe_websocket_send(websocket, {
                'type': 'error',
                'message': 'Claude API key is required for Claude translation provider'
            })
            await safe_websocket_close(websocket, code=4003, reason="Missing Claude API key for translation")
            return
        elif translation_provider == 'gemini' and (not gemini_api_key or gemini_api_key.strip() == ""):
            await safe_websocket_send(websocket, {
                'type': 'error',
                'message': 'Gemini API key is required for Gemini translation provider'
            })
            await safe_websocket_close(websocket, code=4003, reason="Missing Gemini API key for translation")
            return
        
        # Send ready signal to client to start audio streaming immediately
        await safe_websocket_send(websocket, {
            'type': 'ready',
            'message': 'Authentication successful, ready for audio'
        })

        # getting the time and date from the auth_message
        time = auth_message.get('session_time')
        date = auth_message.get('session_date')

    
        # Initialize Assistant
        try:
            assistant = Assistant(
                websocket, 
                dg_api_key=dg_api_key, 
                openai_api_key=openai_api_key, 
                source_language=source_language, 
                target_language=target_language, 
                mode=mode,
                translation_provider=ai_provider,
                claude_api_key=claude_api_key,
                gemini_api_key=gemini_api_key,
                user_email=user_email,
                ai_mode=ai_mode,
                ai_provider=ai_provider,
                web_page_name=web_page_name,
                selected_model=CHAT_GPT_MODEL_BEING_USED,
                Time=time,
                Date=date
            )
            
            
        except Exception as e:
            await safe_websocket_send(websocket, {
                'type': 'error',
                'message': 'Failed to initialize assistant. Please check your API keys and try again.'
            })
            await safe_websocket_close(websocket, code=4004, reason="Assistant initialization failed")
            return
        
        try:
            await asyncio.wait_for(assistant.run(), timeout=21600)  # 6 hours timeout
            
        except asyncio.TimeoutError:
            await safe_websocket_send(websocket, {
                'type': 'error',
                'message': 'Session timeout. Please refresh and start a new session.'
            })
            await safe_websocket_close(websocket, code=4008, reason="Session timeout")
            return
            
        except ConnectionResetError:
            return
            
        except Exception as e:
            await safe_websocket_send(websocket, {
                'type': 'error',
                'message': 'An error occurred during the session. Please try again.'
            })
            await safe_websocket_close(websocket, code=4005, reason="Assistant runtime error")
            return
            
    except ConnectionResetError:
        return
        
    except Exception as e:
        await safe_websocket_send(websocket, {
            'type': 'error',
            'message': 'An unexpected error occurred. Please refresh and try again.'
        })
        await safe_websocket_close(websocket, code=4002, reason="Unexpected error")
        return
        
    finally:
        # Cleanup resources
        if assistant:
            try:
                if hasattr(assistant, 'cleanup'):
                    await assistant.cleanup()
            except Exception as e:
                pass
        
