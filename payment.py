from Autentication import Find_User_DB, collection
from datetime import datetime
import logging
from Packages import Model_table


# Credit cost per minute for each model (in credits) using name and cost like "o3-mini": 0.5
Credit_cost_per_minute = {}
for provider, models in Model_table.items():
    for model, cost in models.items():
        Credit_cost_per_minute[model] = cost

# Default credits for new users
DEFAULT_CREDITS = 10.0
PLATFORM_COST_PER_MINUTE = 0.032
PLATFORM_COST = 0.5

class CreditManager:
    """
    Manages credit operations for users including:
    - Credit balance checking and updating
    - Per-minute deduction based on model
    - Database operations for credit tracking
    - Credit addition functionality
    """
    
    def __init__(self):
        pass
    
    def get_user_credits(self, email):
        """
        Get the current credit balance for a user.
        If no credits field exists, create it with default credits.
        
        Args:
            email (str): User's email address
            
        Returns:
            float: Current credit balance
        """
        try:
            email = email.lower()
            user = Find_User_DB(email)
            
            if not user:
                #self.logger.error(f"User not found: {email}")
                return 0.0
            
            # Check if credits field exists
            if 'Credits' not in user:
                # Initialize with default credits
                # self.logger.info(f"Initializing credits for user {email} with {DEFAULT_CREDITS} credits")
                self.initialize_user_credits(email, DEFAULT_CREDITS)
                return DEFAULT_CREDITS
            
            current_credits = float(user['Credits'])
            # self.logger.debug(f"User {email} has {current_credits} credits")
            return current_credits
            
        except Exception as e:
            # self.logger.error(f"Error getting credits for {email}: {str(e)}")
            return 0.0
    
    def initialize_user_credits(self, email, credits=DEFAULT_CREDITS):
        """
        Initialize credits field for a user.
        
        Args:
            email (str): User's email address
            credits (float): Initial credit amount
            
        Returns:
            bool: Success status
        """
        try:
            email = email.lower()
            result = collection.update_one(
                {"Email": email},
                {"$set": {"Credits": float(credits)}},
                upsert=False
            )
            
            if result.modified_count > 0:
                # self.logger.info(f"Initialized {credits} credits for user {email}")
                return True
            else:
                # self.logger.warning(f"Failed to initialize credits for user {email}")
                return False
                
        except Exception as e:
            #self.logger.error(f"Error initializing credits for {email}: {str(e)}")
            return False
    
    def update_user_credits(self, email, new_balance):
        """
        Update the credit balance for a user in the database.
        
        Args:
            email (str): User's email address
            new_balance (float): New credit balance
            
        Returns:
            bool: Success status
        """
        try:
            email = email.lower()
            result = collection.update_one(
                {"Email": email},
                {"$set": {"Credits": float(new_balance)}},
                upsert=False
            )
            
            if result.modified_count > 0:
                #self.logger.info(f"Updated credits for user {email} to {new_balance}")
                return True
            else:
                #self.logger.warning(f"Failed to update credits for user {email}")
                return False
                
        except Exception as e:
            #self.logger.error(f"Error updating credits for {email}: {str(e)}")
            return False
    
    def deduct_credits(self, email, amount, reason="transcription", model=None, auto_log=True):
        """
        Deduct credits from a user's balance.
        
        Args:
            email (str): User's email address
            amount (float): Amount to deduct
            reason (str): Reason for deduction
            model (str): Model being used (optional)
            auto_log (bool): Whether to automatically log the transaction
            
        Returns:
            tuple: (success, new_balance, message)
        """
        try:
            current_balance = self.get_user_credits(email)
            
            if current_balance < amount:
                #self.logger.warning(f"Insufficient credits for {email}. Required: {amount}, Available: {current_balance}")
                return False, current_balance, f"Insufficient credits. Required: {amount:.2f}, Available: {current_balance:.2f}"
            
            new_balance = current_balance - amount
            
            if self.update_user_credits(email, new_balance):
                # Automatically log the transaction unless disabled
                if auto_log:
                    self.log_credit_transaction(email, "deduct", amount, model, reason)
                
                #self.logger.info(f"Deducted {amount} credits from {email} for {reason}. New balance: {new_balance}")
                return True, new_balance, f"Successfully deducted {amount:.2f} credits"
            else:
                return False, current_balance, "Failed to update credit balance"
                
        except Exception as e:
            #self.logger.error(f"Error deducting credits for {email}: {str(e)}")
            return False, 0.0, f"Error deducting credits: {str(e)}"
    
    def add_credits(self, email, amount, reason="manual_addition", model=None, auto_log=True):
        """
        Add credits to a user's balance.
        
        Args:
            email (str): User's email address
            amount (float): Amount to add
            reason (str): Reason for addition
            model (str): Model being used (optional)
            auto_log (bool): Whether to automatically log the transaction
            
        Returns:
            tuple: (success, new_balance, message)
        """
        try:
            current_balance = self.get_user_credits(email)
            new_balance = current_balance + amount
            
            if self.update_user_credits(email, new_balance):
                # Automatically log the transaction unless disabled
                if auto_log:
                    self.log_credit_transaction(email, "add", amount, model, reason)
                
                #self.logger.info(f"Added {amount} credits to {email} for {reason}. New balance: {new_balance}")
                return True, new_balance, f"Successfully added {amount:.2f} credits"
            else:
                return False, current_balance, "Failed to update credit balance"
                
        except Exception as e:
            #self.logger.error(f"Error adding credits for {email}: {str(e)}")
            return False, 0.0, f"Error adding credits: {str(e)}"
    
    def check_sufficient_credits(self, email, model, duration_minutes=1.0):
        """
        Check if user has sufficient credits for the specified duration and model.
        
        Args:
            email (str): User's email address
            model (str): Model being used
            duration_minutes (float): Duration in minutes
            
        Returns:
            tuple: (sufficient, current_balance, required_credits)
        """
        try:
            current_balance = self.get_user_credits(email)
            
            # Use the new total cost calculation
            cost_per_minute = self.calculate_total_cost_per_minute(model)
            required_credits = cost_per_minute * duration_minutes
            sufficient = current_balance >= required_credits
            
            #self.logger.debug(f"Credit check for {email}: Balance={current_balance}, Required={required_credits}, Sufficient={sufficient}")
            
            return sufficient, current_balance, required_credits
            
        except Exception as e:
            #self.logger.error(f"Error checking credits for {email}: {str(e)}")
            return False, 0.0, 0.0
    
    def get_model_cost_per_minute(self, model):
        """
        Get the cost per minute for a specific model.
        
        Args:
            model (str): Model name
            
        Returns:
            float: Cost per minute in credits
        """
        return Credit_cost_per_minute.get(model, 0.5)  # Default to 0.5 if model not found
    
    def Calculate_Cost_of_Premium_Feature(self, model, number_of_words,Summary_output,Keyword_output):
        """
        Calculate the cost of a premium feature.
        
        Args:
            model (str): Model name
            number_of_words (int): Number of words
            
        Returns:
            float: Cost of the premium feature in credits
        """

        Summary_Prompt_words = 720
        Keyword_Prompt_words = 150
        
        Cost = ((((number_of_words + Summary_Prompt_words + Keyword_Prompt_words) * 1.5)/1000000) * self.get_model_cost_per_minute(model)) + ((((Summary_output + Keyword_output)*1.5)/1000000) * self.get_model_cost_per_minute(model))

        Cost += PLATFORM_COST

        return Cost
        


    
    
    def calculate_total_cost_per_minute(self, model):
        """
        Calculate the total cost per minute including AI model cost, Deepgram cost, and platform cost.
        
        Formula: Total Cost = Model Cost + Deepgram Cost + Platform Cost
        Where:
        - Model Cost: Variable based on AI model used
        - Deepgram Cost: 0.0043 credits per minute
        - Platform Cost: 0.032 credits per minute
        
        Args:
            model (str): Model name
            
        Returns:
            float: Total cost per minute in credits
        """
        # Constants
        DEEPGRAM_COST_PER_MINUTE = 0.0043
        
        # Get the base model cost
        model_cost = self.get_model_cost_per_minute(model)
        
        # Calculate total cost
        total_cost = model_cost + DEEPGRAM_COST_PER_MINUTE + PLATFORM_COST_PER_MINUTE
        
        #self.logger.debug(f"Cost breakdown for {model}: Model={model_cost}, Deepgram={DEEPGRAM_COST_PER_MINUTE}, Platform={PLATFORM_COST_PER_MINUTE}, Total={total_cost}")
        
        return total_cost
    
    def calculate_session_cost(self, model, duration_minutes):
        """
        Calculate the total cost for a session.
        
        Args:
            model (str): Model being used
            duration_minutes (float): Duration in minutes
            
        Returns:
            float: Total cost in credits
        """
        cost_per_minute = self.get_model_cost_per_minute(model)
        return cost_per_minute * duration_minutes
    
    def log_credit_transaction(self, email, transaction_type, amount, model=None, reason=None, session_id=None, duration_minutes=None):
        """
        Log a credit transaction for audit purposes.
        
        Args:
            email (str): User's email address
            transaction_type (str): 'deduct', 'add', or 'purchase'
            amount (float): Amount of credits
            model (str): Model used (if applicable)
            reason (str): Reason for transaction
            session_id (str): Session identifier (optional)
            duration_minutes (float): Session duration in minutes (if applicable)
        """
        try:
            email = email.lower()
            transaction = {
                "timestamp": datetime.now().isoformat(),
                "type": transaction_type,
                "amount": float(amount),
                "model": model,
                "reason": reason,
                "session_id": session_id,
                "duration_minutes": duration_minutes
            }
            
            # Remove None values to keep the transaction clean
            transaction = {k: v for k, v in transaction.items() if v is not None}
            
            # Add to user's transaction log
            collection.update_one(
                {"Email": email},
                {"$push": {"CreditTransactions": transaction}},
                upsert=False
            )
            
            #self.logger.info(f"Logged credit transaction for {email}: {transaction}")
            
        except Exception as e:
            #self.logger.error(f"Error logging credit transaction for {email}: {str(e)}")
            pass
    
    def log_session_credits(self, email, total_credits_used, model, session_duration, platform, session_id=None, time=None, date=None):
        """
        Log a comprehensive session-level credit transaction.
        
        Args:
            email (str): User's email address
            total_credits_used (float): Total credits used in the session
            model (str): Model used for the session
            session_duration (float): Session duration in minutes
            platform (str): Platform/website name
            session_id (str): Session identifier (optional)
        """
        try:
            if total_credits_used <= 0:
                #self.logger.info(f"No credits to log for session {session_id or 'unknown'}")
                return
            
            session_reason = f"session_{platform}_{session_duration:.2f}min_{model}"
            
            self.log_credit_transaction(
                email=email,
                transaction_type="deduct",
                amount=total_credits_used,
                model=model,
                reason=session_reason,
                session_id=session_id,
                duration_minutes=session_duration
            )
            
            #self.logger.info(f"Session credit transaction logged for {email}: {total_credits_used:.3f} credits, {session_duration:.2f} minutes, model: {model}")
            
        except Exception as e:
            #self.logger.error(f"Error logging session credits for {email}: {str(e)}")
            pass
    
    def get_credit_history(self, email, page=1, limit=10):
        """
        Get credit transaction history for a user with pagination.
        
        Args:
            email (str): User's email address
            page (int): Page number (1-based)
            limit (int): Number of transactions per page
            
        Returns:
            dict: Transaction history with pagination info
        """
        try:
            email = email.lower()
            user = Find_User_DB(email)
            
            if not user:
                return {
                    'transactions': [],
                    'transaction_count': 0,
                    'current_credits': 0,
                    'total_pages': 0
                }
            
            transactions = user.get('CreditTransactions', [])
            current_credits = user.get('Credits', 0)
            
            # Sort transactions by timestamp (newest first)
            transactions = sorted(transactions, key=lambda x: x.get('timestamp', ''), reverse=True)
            
            # Calculate pagination
            total_transactions = len(transactions)
            total_pages = (total_transactions + limit - 1) // limit  # Ceiling division
            start_idx = (page - 1) * limit
            end_idx = start_idx + limit
            
            paginated_transactions = transactions[start_idx:end_idx]
            
            return {
                'transactions': paginated_transactions,
                'transaction_count': total_transactions,
                'current_credits': current_credits,
                'total_pages': total_pages
            }
            
        except Exception as e:
            #self.logger.error(f"Error getting credit history for {email}: {str(e)}")
            return {
                'transactions': [],
                'transaction_count': 0,
                'current_credits': 0,
                'total_pages': 0
            }

# Global instance
credit_manager = CreditManager()  