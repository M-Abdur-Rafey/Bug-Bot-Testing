import stripe
import os
import logging
from fastapi import HTTPException
from dotenv import load_dotenv
from payment import CreditManager
from Autentication import Find_User_DB, collection
from Packages import Packages
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

#logger = logging.get#logger(__name__)
credit_manager = CreditManager()

class StripePaymentManager:
    """
    Manages Stripe payment operations including:
    - Payment verification
    - Webhook handling
    - Credit allocation after successful payments
    - Payment link creation
    - Frontend-specific payment flows
    - Customer management
    """
    
    def __init__(self):
        self.stripe = stripe
        self.webhook_secret = STRIPE_WEBHOOK_SECRET
        
    def verify_webhook_signature(self, payload: bytes, signature: str):
        """
        Verify the Stripe webhook signature to ensure the request is from Stripe.
        
        Args:
            payload (bytes): The raw request body
            signature (str): The Stripe signature header
            
        Returns:
            dict: The verified event data
            
        Raises:
            HTTPException: If signature verification fails
        """
        try:
            event = stripe.Webhook.construct_event(
                payload, signature, self.webhook_secret
            )
            return event
        except ValueError as e:
            #logger.error(f"Invalid payload: {e}")
            raise HTTPException(status_code=400, detail="Invalid payload")
        except stripe.error.SignatureVerificationError as e:
            #logger.error(f"Invalid signature: {e}")
            raise HTTPException(status_code=400, detail="Invalid signature")
    
    def handle_payment_success(self, event_data: dict):
        """
        Handle successful payment events and allocate credits to users.
        Enhanced to handle multiple payment scenarios, customer management, better email detection,
        and idempotency protection to prevent duplicate credit additions.
        
        Args:
            event_data (dict): The Stripe event data
            
        Returns:
            dict: Processing result
        """
        try:
            # Get the payment intent or checkout session
            payment_intent = None
            customer_email = None
            customer_name = None
            amount = 0
            metadata = {}
            session_id = None
            payment_intent_id = None
            stripe_customer_id = None
            
            if event_data['type'] == 'payment_intent.succeeded':
                payment_intent = event_data['data']['object']
                payment_intent_id = payment_intent['id']
                customer_email = payment_intent.get('receipt_email')
                amount = payment_intent.get('amount_received', 0) / 100  # Convert from cents
                metadata = payment_intent.get('metadata', {})
                stripe_customer_id = payment_intent.get('customer')
                
            elif event_data['type'] == 'checkout.session.completed':
                session = event_data['data']['object']
                session_id = session['id']
                customer_email = session.get('customer_details', {}).get('email')
                customer_name = session.get('customer_details', {}).get('name')
                amount = session.get('amount_total', 0) / 100  # Convert from cents
                metadata = session.get('metadata', {})
                stripe_customer_id = session.get('customer')
                
                # If checkout session has payment_intent, get email from there too
                if not customer_email and session.get('payment_intent'):
                    payment_intent_id = session.get('payment_intent')
                    try:
                        pi = stripe.PaymentIntent.retrieve(payment_intent_id)
                        customer_email = pi.get('receipt_email') or metadata.get('customer_email')
                        if not stripe_customer_id:
                            stripe_customer_id = pi.get('customer')
                    except Exception as e:
                        #logger.warning(f"Could not retrieve payment intent {payment_intent_id}: {e}")
                        pass
                        
            else:
                #logger.warning(f"Unhandled event type: {event_data['type']}")
                return {"status": "ignored", "message": "Event type not handled"}
            
            # Try to get email from metadata if not found elsewhere
            if not customer_email:
                customer_email = metadata.get('customer_email')
            
            if not customer_email:
                #logger.error("No customer email found in payment data")
                return {"status": "error", "message": "No customer email found"}
            
            # Normalize email
            customer_email = customer_email.lower()
            
            # IDEMPOTENCY CHECK: Verify this payment hasn't already been processed
            #logger.info(f"Checking for duplicate payment processing - Session: {session_id}, Payment Intent: {payment_intent_id}")
            
            existing_user = collection.find_one({"Email": customer_email})
            if existing_user:
                credit_transactions = existing_user.get('CreditTransactions', [])
                
                # Check if we already have a transaction for this session ID or payment intent
                for transaction in credit_transactions:
                    if ((session_id and transaction.get('stripe_session_id') == session_id) or 
                        (payment_intent_id and transaction.get('stripe_payment_intent_id') == payment_intent_id)):
                        #logger.warning(f"Payment already processed via webhook - Session: {session_id}, Payment Intent: {payment_intent_id}")
                        return {
                            "status": "duplicate", 
                            "message": f"Payment already processed for {customer_email}",
                            "transaction_id": session_id or payment_intent_id
                        }
            
            # Handle Stripe Customer ID - create or get existing customer
            if not stripe_customer_id:
                #logger.info(f"No customer ID found in payment, creating/getting customer for {customer_email}")
                customer_result = self.create_or_get_customer(customer_email, customer_name)
                if customer_result['status'] == 'success':
                    stripe_customer_id = customer_result['customer_id']
                    #logger.info(f"Using customer ID: {stripe_customer_id} for {customer_email}")
                else:
                    #logger.warning(f"Could not create/get customer for {customer_email}: {customer_result.get('message')}")
                    pass
            # Save customer ID to user database (only if we have one)
            if stripe_customer_id:
                saved = self.save_customer_id_to_user(customer_email, stripe_customer_id)
                if saved:
                    #logger.info(f"Saved Stripe Customer ID {stripe_customer_id} for user {customer_email}")
                    pass
                else:
                    #logger.warning(f"Could not save Stripe Customer ID for user {customer_email}")
                    pass
            # Determine package based on amount
            package_info = self._get_package_by_amount(amount)
            if not package_info:
                #logger.error(f"No package found for amount: {amount}")
                return {"status": "error", "message": "Invalid payment amount"}
            
            # Add credits to user account
            success, new_balance, message = credit_manager.add_credits(
                customer_email,
                package_info['Credits'],
                f"Stripe webhook - {package_info['name']} package (Session: {session_id or payment_intent_id})",
                auto_log=False  # We'll log manually with enhanced details
            )
            
            if success:
                # Update user's paid status
                self._update_user_payment_status(customer_email, package_info)
                
                # Log enhanced transaction details to CreditTransactions
                transaction_details = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "purchase",
                    "amount": float(package_info['Credits']),
                    "reason": f"Stripe webhook - {package_info['name']} package (Session: {session_id or payment_intent_id})",
                    "stripe_session_id": session_id,
                    "stripe_payment_intent_id": payment_intent_id,
                    "stripe_customer_id": stripe_customer_id,  # Added customer ID
                    "package_name": package_info['name'],
                    "package_price": package_info['Price'],
                    "payment_currency": "USD",
                    "payment_method": "stripe",
                    "payment_status": "completed",
                    "invoice_data": {
                        "transaction_id": session_id or payment_intent_id,
                        "customer_email": customer_email,
                        "stripe_customer_id": stripe_customer_id,
                        "amount_paid": amount,
                        "credits_purchased": package_info['Credits'],
                        "package_description": package_info['Description'],
                        "payment_date": datetime.utcnow().isoformat()
                    }
                }
                
                # Add enhanced transaction to user's record
                collection.update_one(
                    {"Email": customer_email},
                    {"$push": {"CreditTransactions": transaction_details}},
                    upsert=False
                )
                
                #logger.info(f"Successfully processed webhook payment: Added {package_info['Credits']} credits to {customer_email}")
                return {
                    "status": "success",
                    "message": f"Added {package_info['Credits']} credits to {customer_email}",
                    "package": package_info['name'],
                    "new_balance": new_balance,
                    "transaction_id": session_id or payment_intent_id,
                    "customer_id": stripe_customer_id
                }
            else:
                #logger.error(f"Failed to add credits to {customer_email}: {message}")
                return {"status": "error", "message": message}
                
        except Exception as e:
            #logger.error(f"Error handling payment success: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _get_package_by_amount(self, amount: float):
        """
        Get package information based on payment amount.
        Enhanced to handle both integer and float amounts.
        
        Args:
            amount (float): Payment amount in dollars
            
        Returns:
            dict: Package information or None if not found
        """
        for package_name, package_data in Packages.items():
            if abs(package_data['Price'] - amount) < 0.01:  # Allow for small floating point differences
                return {
                    'name': package_name,
                    'Credits': package_data['Credits'],
                    'Price': package_data['Price'],
                    'Description': package_data['Description']
                }
        return None
    
    def _update_user_payment_status(self, email: str, package_info: dict):
        """
        Update user's payment status in the database.
        
        Args:
            email (str): User's email
            package_info (dict): Package information
        """
        try:
            # Update user as paid user and log the transaction
            update_data = {
                'paid_user': True,
                'Paid_User': True,  # Match the database field name
                'last_payment_date': datetime.utcnow().isoformat(),
                'last_package': package_info['name']
            }
            
            collection.update_one(
                {"Email": email.lower()},
                {"$set": update_data}
            )
            
            # Log the credit transaction with enhanced details
            credit_manager.log_credit_transaction(
                email,
                "purchase", 
                package_info['Credits'],
                reason=f"Stripe payment - {package_info['name']} package"
            )
            
        except Exception as e:
            #logger.error(f"Error updating user payment status: {str(e)}")
            pass
    
    def create_payment_intent_for_credits(self, email: str, credits: int, price: float):
        """
        Create a Stripe payment intent specifically for credit purchases from the frontend.
        
        Args:
            email (str): Customer email
            credits (int): Number of credits being purchased
            price (float): Price in dollars
            
        Returns:
            dict: Payment intent data
        """
        try:
            amount_cents = int(price * 100)  # Convert to cents
            
            intent = stripe.PaymentIntent.create(
                amount=amount_cents,
                currency="usd",
                metadata={
                    'customer_email': email,
                    'credits': str(credits),
                    'package_type': 'credit_purchase'
                },
                receipt_email=email,
                description=f"Credit Purchase: {credits} credits for {email}"
            )
            
            return {
                "status": "success", 
                "client_secret": intent.client_secret,
                "payment_intent_id": intent.id
            }
        except Exception as e:
            #logger.error(f"Error creating payment intent for credits: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def create_checkout_session_for_package(self, email: str, package_name: str, return_url: str, cancel_url: str):
        """
        Create a Stripe checkout session for a specific package.
        Now creates or gets existing Stripe customer to ensure customer ID is available.
        
        Args:
            email (str): Customer email
            package_name (str): Name of the package (Basic, Pro, Enterprise)
            return_url (str): URL to redirect to after successful payment
            cancel_url (str): URL to redirect to if payment is canceled
            
        Returns:
            dict: Checkout session data
        """
        try:
            if package_name not in Packages:
                return {"status": "error", "message": "Invalid package name"}
            
            package = Packages[package_name]
            
            # Create or get existing Stripe customer
            customer_result = self.create_or_get_customer(email)
            if customer_result['status'] != 'success':
                #logger.warning(f"Could not create/get customer for {email}, proceeding with customer_email only")
                customer_id = None
            else:
                customer_id = customer_result['customer_id']
                #logger.info(f"Using customer ID {customer_id} for checkout session")
            
            # Update success_url to redirect to payment-details with session_id
            # Note: {CHECKOUT_SESSION_ID} will be replaced by Stripe with actual session ID
            success_url_with_session = f"{return_url}?session_id={{CHECKOUT_SESSION_ID}}&package={package_name}&credits={package['Credits']}&amount={package['Price']}"
            
            #logger.info(f"Creating checkout session with success_url: {success_url_with_session}")
            
            # Create session data
            session_data = {
                'payment_method_types': ['card'],
                'line_items': [{
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': f'{package_name} Package - {package["Credits"]} Credits',
                            'description': package['Description']
                        },
                        'unit_amount': int(package['Price'] * 100),  # Convert to cents
                    },
                    'quantity': 1,
                }],
                'mode': 'payment',
                'success_url': success_url_with_session,
                'cancel_url': cancel_url,
                'metadata': {
                    'customer_email': email,
                    'package_name': package_name,
                    'credits': str(package['Credits'])
                }
            }
            
            # Use customer ID if available, otherwise use customer_email
            if customer_id:
                session_data['customer'] = customer_id
            else:
                session_data['customer_email'] = email
            
            session = stripe.checkout.Session.create(**session_data)
            
            return {
                "status": "success", 
                "checkout_url": session.url,
                "session_id": session.id,
                "customer_id": customer_id
            }
        except Exception as e:
            #logger.error(f"Error creating checkout session: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def create_payment_intent(self, amount: int, currency: str = "usd", metadata: dict = None):
        """
        Create a Stripe payment intent.
        
        Args:
            amount (int): Amount in cents
            currency (str): Currency code
            metadata (dict): Additional metadata
            
        Returns:
            dict: Payment intent data
        """
        try:
            intent = stripe.PaymentIntent.create(
                amount=amount,
                currency=currency,
                metadata=metadata or {}
            )
            return {"status": "success", "client_secret": intent.client_secret}
        except Exception as e:
            #logger.error(f"Error creating payment intent: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def retrieve_payment_status(self, payment_intent_id: str):
        """
        Retrieve the status of a payment intent.
        
        Args:
            payment_intent_id (str): The payment intent ID
            
        Returns:
            dict: Payment status information
        """
        try:
            intent = stripe.PaymentIntent.retrieve(payment_intent_id)
            return {
                "status": "success",
                "payment_status": intent.status,
                "amount": intent.amount,
                "currency": intent.currency,
                "metadata": intent.metadata
            }
        except Exception as e:
            #logger.error(f"Error retrieving payment status: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def handle_payment_failed(self, event_data: dict):
        """
        Handle failed payment events.
        
        Args:
            event_data (dict): The Stripe event data
            
        Returns:
            dict: Processing result
        """
        try:
            if event_data['type'] == 'payment_intent.payment_failed':
                payment_intent = event_data['data']['object']
                customer_email = payment_intent.get('receipt_email') or payment_intent.get('metadata', {}).get('customer_email')
                
                #logger.warning(f"Payment failed for customer: {customer_email}")
                
                # You can add additional logic here like sending failure notifications
                return {
                    "status": "handled",
                    "message": f"Payment failure logged for {customer_email}"
                }
            
            return {"status": "ignored", "message": "Event type not handled"}
            
        except Exception as e:
            #logger.error(f"Error handling payment failure: {str(e)}")
            return {"status": "error", "message": str(e)}

    def create_or_get_customer(self, email: str, name: str = None):
        """
        Create a new Stripe customer or get existing one by email.
        
        Args:
            email (str): Customer email
            name (str): Customer name (optional)
            
        Returns:
            dict: Customer data with customer_id
        """
        try:
            # First, try to find existing customer by email
            customers = stripe.Customer.list(email=email, limit=1)
            
            if customers.data:
                # Customer exists, return existing customer
                customer = customers.data[0]
                #logger.info(f"Found existing Stripe customer: {customer.id} for email: {email}")
                return {
                    "status": "success",
                    "customer_id": customer.id,
                    "customer": customer,
                    "created": False
                }
            else:
                # Create new customer
                customer_data = {"email": email}
                if name:
                    customer_data["name"] = name
                
                customer = stripe.Customer.create(**customer_data)
                #logger.info(f"Created new Stripe customer: {customer.id} for email: {email}")
                
                return {
                    "status": "success",
                    "customer_id": customer.id,
                    "customer": customer,
                    "created": True
                }
                
        except Exception as e:
            #logger.error(f"Error creating/getting Stripe customer for {email}: {str(e)}")
            return {"status": "error", "message": str(e)}

    def get_customer_by_id(self, customer_id: str):
        """
        Retrieve a Stripe customer by ID.
        
        Args:
            customer_id (str): Stripe customer ID
            
        Returns:
            dict: Customer data
        """
        try:
            customer = stripe.Customer.retrieve(customer_id)
            return {
                "status": "success",
                "customer": customer
            }
        except Exception as e:
            #logger.error(f"Error retrieving Stripe customer {customer_id}: {str(e)}")
            return {"status": "error", "message": str(e)}

    def save_customer_id_to_user(self, email: str, stripe_customer_id: str):
        """
        Save Stripe Customer ID to user database.
        
        Args:
            email (str): User email
            stripe_customer_id (str): Stripe customer ID
            
        Returns:
            bool: Success status
        """
        try:
            from User import User
            user = User()
            return user.Update_Stripe_Customer_ID(email, stripe_customer_id)
        except Exception as e:
            #logger.error(f"Error saving customer ID to user {email}: {str(e)}")
            return False

    def get_transaction_details_from_stripe(self, identifier: str):
        """
        Retrieve transaction details directly from Stripe using various identifiers.
        
        Args:
            identifier (str): Can be customer ID (cus_), session ID (cs_), or payment intent ID (pi_)
            
        Returns:
            dict: Transaction details or error
        """
        try:
            #logger.info(f"Retrieving transaction details from Stripe for identifier: {identifier}")
            
            if identifier.startswith('cus_'):
                # Customer ID - get recent payment intents for this customer
                return self._get_customer_transactions(identifier)
                
            elif identifier.startswith('cs_'):
                # Checkout session ID
                return self._get_session_transaction(identifier)
                
            elif identifier.startswith('pi_'):
                # Payment intent ID
                return self._get_payment_intent_transaction(identifier)
                
            else:
                return {"status": "error", "message": "Invalid identifier format"}
                
        except Exception as e:
            #logger.error(f"Error retrieving transaction from Stripe: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _get_customer_transactions(self, customer_id: str):
        """Get recent transactions for a customer."""
        try:
            # Get recent payment intents for this customer
            payment_intents = stripe.PaymentIntent.list(
                customer=customer_id,
                limit=10,
                expand=['data.invoice']
            )
            
            if not payment_intents.data:
                return {"status": "error", "message": "No transactions found for customer"}
            
            # Get the most recent successful payment
            for pi in payment_intents.data:
                if pi.status == 'succeeded':
                    transaction_data = self._format_transaction_data(pi, pi.id, customer_id)
                    return {"status": "success", "transaction": transaction_data}
            
            return {"status": "error", "message": "No successful transactions found"}
            
        except Exception as e:
            #logger.error(f"Error getting customer transactions: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _get_session_transaction(self, session_id: str):
        """Get transaction details from checkout session."""
        try:
            session = stripe.checkout.Session.retrieve(
                session_id,
                expand=['payment_intent', 'subscription', 'line_items']
            )
            
            if session.payment_status != 'paid':
                return {"status": "error", "message": "Session payment not completed"}
            
            payment_intent_id = session.payment_intent
            customer_id = session.customer
            
            # Get payment intent details if available
            payment_intent = None
            if payment_intent_id:
                if isinstance(payment_intent_id, str):
                    payment_intent = stripe.PaymentIntent.retrieve(payment_intent_id)
                else:
                    payment_intent = payment_intent_id
            
            transaction_data = self._format_session_transaction_data(session, payment_intent, customer_id)
            return {"status": "success", "transaction": transaction_data}
            
        except Exception as e:
            #logger.error(f"Error getting session transaction: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _get_payment_intent_transaction(self, payment_intent_id: str):
        """Get transaction details from payment intent."""
        try:
            payment_intent = stripe.PaymentIntent.retrieve(
                payment_intent_id,
                expand=['invoice']
            )
            
            if payment_intent.status != 'succeeded':
                return {"status": "error", "message": "Payment intent not succeeded"}
            
            customer_id = payment_intent.customer
            transaction_data = self._format_transaction_data(payment_intent, payment_intent_id, customer_id)
            return {"status": "success", "transaction": transaction_data}
            
        except Exception as e:
            #logger.error(f"Error getting payment intent transaction: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _format_transaction_data(self, payment_intent, payment_intent_id: str, customer_id: str):
        """Format transaction data in a consistent format."""
        amount = payment_intent.amount / 100  # Convert from cents
        
        # Get package info based on amount
        package_info = self._get_package_by_amount(amount)
        
        return {
            "timestamp": datetime.fromtimestamp(payment_intent.created).isoformat(),
            "type": "purchase",
            "amount": float(package_info['Credits']) if package_info else 0,
            "reason": f"Stripe payment - {package_info['name']} package" if package_info else "Stripe payment",
            "stripe_session_id": None,  # Will be set if available
            "stripe_payment_intent_id": payment_intent_id,
            "stripe_customer_id": customer_id,
            "package_name": package_info['name'] if package_info else "Unknown",
            "package_price": package_info['Price'] if package_info else amount,
            "payment_currency": payment_intent.currency.upper(),
            "payment_method": "stripe",
            "payment_status": "completed",
            "invoice_data": {
                "transaction_id": payment_intent_id,
                "customer_email": payment_intent.receipt_email or "",
                "stripe_customer_id": customer_id,
                "amount_paid": amount,
                "credits_purchased": package_info['Credits'] if package_info else 0,
                "package_description": package_info['Description'] if package_info else "",
                "payment_date": datetime.fromtimestamp(payment_intent.created).isoformat()
            }
        }

    def _format_session_transaction_data(self, session, payment_intent, customer_id: str):
        """Format session transaction data in a consistent format."""
        amount = session.amount_total / 100 if session.amount_total else 0
        
        # Get package info based on amount
        package_info = self._get_package_by_amount(amount)
        
        # Get customer email from session or payment intent
        customer_email = ""
        if session.customer_details and session.customer_details.email:
            customer_email = session.customer_details.email
        elif payment_intent and payment_intent.receipt_email:
            customer_email = payment_intent.receipt_email
        
        return {
            "timestamp": datetime.fromtimestamp(session.created).isoformat(),
            "type": "purchase",
            "amount": float(package_info['Credits']) if package_info else 0,
            "reason": f"Stripe payment - {package_info['name']} package" if package_info else "Stripe payment",
            "stripe_session_id": session.id,
            "stripe_payment_intent_id": payment_intent.id if payment_intent else None,
            "stripe_customer_id": customer_id,
            "package_name": package_info['name'] if package_info else "Unknown",
            "package_price": package_info['Price'] if package_info else amount,
            "payment_currency": session.currency.upper() if session.currency else "USD",
            "payment_method": "stripe",
            "payment_status": "completed",
            "invoice_data": {
                "transaction_id": session.id,
                "customer_email": customer_email,
                "stripe_customer_id": customer_id,
                "amount_paid": amount,
                "credits_purchased": package_info['Credits'] if package_info else 0,
                "package_description": package_info['Description'] if package_info else "",
                "payment_date": datetime.fromtimestamp(session.created).isoformat()
            }
        }

    def save_transaction_to_database(self, email: str, transaction_data: dict):
        """
        Save transaction details to user's database.
        
        Args:
            email (str): User email
            transaction_data (dict): Transaction data to save
            
        Returns:
            bool: Success status
        """
        try:
            # Check if transaction already exists to avoid duplicates
            user = Find_User_DB(email.lower())
            if not user:
                #logger.error(f"User not found: {email}")
                return False
            
            # Handle corrupted CreditTransactions field - ensure it's always an array
            existing_transactions = user.get('CreditTransactions', [])
            
            # Check if CreditTransactions is corrupted (stored as object instead of array)
            if not isinstance(existing_transactions, list):
                #logger.warning(f"CreditTransactions field is corrupted for user {email}, fixing it...")
                # Reset to empty array and update in database
                collection.update_one(
                    {"Email": email.lower()},
                    {"$set": {"CreditTransactions": []}},
                    upsert=False
                )
                existing_transactions = []
                #logger.info(f"Reset CreditTransactions to empty array for user {email}")
            
            # Check for duplicates based on stripe identifiers
            session_id = transaction_data.get('stripe_session_id')
            payment_intent_id = transaction_data.get('stripe_payment_intent_id')
            
            for existing_trans in existing_transactions:
                if (session_id and existing_trans.get('stripe_session_id') == session_id) or \
                   (payment_intent_id and existing_trans.get('stripe_payment_intent_id') == payment_intent_id):
                    #logger.info(f"Transaction already exists in database for {email}")
                    return True
            
            # Add transaction to user's record
            collection.update_one(
                {"Email": email.lower()},
                {"$push": {"CreditTransactions": transaction_data}},
                upsert=False
            )
            
            #logger.info(f"Successfully saved transaction to database for {email}")
            return True
            
        except Exception as e:
            #logger.error(f"Error saving transaction to database: {str(e)}")
            # If there's still an error, try to fix the database schema
            try:
                #logger.info(f"Attempting to fix database schema for user {email}")
                collection.update_one(
                    {"Email": email.lower()},
                    {"$set": {"CreditTransactions": [transaction_data]}},
                    upsert=False
                )
                #logger.info(f"Successfully fixed database schema and saved transaction for {email}")
                return True
            except Exception as schema_error:
                #logger.error(f"Failed to fix database schema: {str(schema_error)}")
                return False

    def fix_user_credit_transactions(self, email: str):
        """
        Fix corrupted CreditTransactions field for a specific user.
        
        Args:
            email (str): User email
            
        Returns:
            bool: Success status
        """
        try:
            user = Find_User_DB(email.lower())
            if not user:
                #logger.error(f"User not found: {email}")
                return False
            
            existing_transactions = user.get('CreditTransactions', [])
            
            if not isinstance(existing_transactions, list):
                #logger.info(f"Fixing corrupted CreditTransactions for user {email}")
                # Reset to empty array
                collection.update_one(
                    {"Email": email.lower()},
                    {"$set": {"CreditTransactions": []}},
                    upsert=False
                )
                #logger.info(f"Successfully fixed CreditTransactions for user {email}")
                return True
            else:
                #logger.info(f"CreditTransactions is already in correct format for user {email}")
                return True
                
        except Exception as e:
            #logger.error(f"Error fixing CreditTransactions for user {email}: {str(e)}")
            return False

    def fix_all_corrupted_credit_transactions(self):
        """
        Fix corrupted CreditTransactions fields for all users in the database.
        This should be run as a maintenance operation.
        
        Returns:
            dict: Repair results
        """
        try:
            results = {
                "total_users_checked": 0,
                "users_fixed": 0,
                "errors": []
            }
            
            # Find all users with corrupted CreditTransactions field
            cursor = collection.find({"CreditTransactions": {"$exists": True, "$not": {"$type": "array"}}})
            
            for user in cursor:
                results["total_users_checked"] += 1
                email = user.get('Email', '')
                
                try:
                    collection.update_one(
                        {"_id": user["_id"]},
                        {"$set": {"CreditTransactions": []}},
                        upsert=False
                    )
                    results["users_fixed"] += 1
                    #logger.info(f"Fixed CreditTransactions for user: {email}")
                    
                except Exception as user_error:
                    error_msg = f"Failed to fix user {email}: {str(user_error)}"
                    results["errors"].append(error_msg)
                    #logger.error(error_msg)
            
            #logger.info(f"Database repair completed: {results}")
            return results
            
        except Exception as e:
            #logger.error(f"Error during database repair: {str(e)}")
            return {"error": str(e)}

# Global instance
stripe_manager = StripePaymentManager()

def get_stripe_manager():
    """Get the global Stripe manager instance."""
    return stripe_manager
