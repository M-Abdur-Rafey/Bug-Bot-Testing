from Autentication import *
from datetime import datetime
import logging

#logger = logging.get#logger(__name__)

class User:
    def __init__(self):
        self.email = None
        self.password = None
        self.name = None
        self.api_key = {}
        self.sessions = {}
        self.Paid_User = False
        self.time = None
        self.date = None
        # #logger.info("User module initialized")
      
    def Find_User(self, email):
        # Convert email to lowercase for consistency
        email = email.lower()
        user = Find_User_DB(email)
        if user:
            return True
        else:
            return False

    def Set_data(self, email, password, name,keys,sessions,Summary_and_Keywords, time, date):
        self.email = email
        self.password = password
        self.name = name
        self.api_key = keys
        self.sessions = sessions
        self.Summary_and_Keywords = Summary_and_Keywords
        self.time = time
        self.date = date
    
    def Get_data(self):
        return self.email, self.password, self.name, self.api_key, self.sessions, self.Summary_and_Keywords, self.time, self.date

    def Update_Data(self):
        # Convert email to lowercase for consistency
        email = self.email.lower() if self.email else ""
        collection.update_one({"Email": email}, {"$set": {"Session": self.sessions}})

    def Get_Session(self,date):
        return self.sessions[date]
    
    def Get_All_Sessions(self):
        return self.sessions
    
    def Get_All_Sessions_Dates(self):
        return list(self.sessions.keys())
    
    def Get_User_Data(self, email):
        # Convert email to lowercase for consistency
        email = email.lower()
        return Find_User_DB(email)

    def Check_User_Credentials(self, email, password):
        # Convert email to lowercase for consistency
        email = email.lower()
        return Authenticate_User(email, password)

    def Add_User(self, user):
        # Convert all user fields to lowercase for consistency
        if isinstance(user, dict):
            user = {key.lower(): value for key, value in user.items()}
            # Ensure email is lowercase
            if 'email' in user:
                user['email'] = user['email'].lower()
        return Add_User_To_DB(user)
    
    def Get_User_Dates(self, email):
        ''' Gets all the dates of the user's sessions '''
        user = self.Get_User_Data(email)
        if user:
            return list(user['Session'].keys())
        else:
            return []
    
    def Get_User_Data_By_Date(self, email, date):
        ''' Gets the data of the user's session for a specific date '''
        user = self.Get_User_Data(email)
        if user:
            return user['Session'][date]
        else:
            return None
        
    def Update_S_and_K(self, email, Summary_and_Keywords):
        ''' Updates the summary and keywords preference of the user '''
        try:
            # Convert email to lowercase for consistency
            email = email.lower()
            
            result = Update_User_Keyword_and_Summary(email, Summary_and_Keywords)
            return result
        except Exception as e:
            return False
        
    def Update_API_Key(self, email, api_key):
        ''' Updates the API key of the user '''
        try:
            # Convert email to lowercase for consistency
            email = email.lower()
            
            user = self.Get_User_Data(email)
            if user:
                # Use upsert to create the field if it doesn't exist
                result = collection.update_one(
                    {"Email": email}, 
                    {"$set": {"API_Key": api_key}}, 
                    upsert=True
                )
                return True
            else:
                return False
        except Exception as e:
            #logger.error(f"Error updating API key for {email}: {str(e)}")
            return False

    def Update_Paid_User(self, email, paid_user_status):
        ''' Updates the Paid User status of the user '''
        try:
            # Convert email to lowercase for consistency
            email = email.lower()
            
            user = self.Get_User_Data(email)
            if user:
                # Use upsert to create the field if it doesn't exist
                result = collection.update_one(
                    {"Email": email}, 
                    {"$set": {"Paid_User": paid_user_status}}, 
                    upsert=True
                )
                #logger.info(f"Updated Paid_User status for {email} to {paid_user_status}")
                return True
            else:
                #logger.error(f"User not found when updating Paid_User status: {email}")
                return False
        except Exception as e:
            #logger.error(f"Error updating Paid_User status for {email}: {str(e)}")
            return False

    def Update_Stripe_Customer_ID(self, email, stripe_customer_id):
        ''' Updates the Stripe Customer ID of the user '''
        try:
            # Convert email to lowercase for consistency
            email = email.lower()
            
            user = self.Get_User_Data(email)
            if user:
                # Use upsert to create the field if it doesn't exist
                result = collection.update_one(
                    {"Email": email}, 
                    {"$set": {"Stripe_Customer_ID": stripe_customer_id}}, 
                    upsert=True
                )
                #logger.info(f"Updated Stripe Customer ID for {email} to {stripe_customer_id}")
                return True
            else:
                #logger.error(f"User not found when updating Stripe Customer ID: {email}")
                return False
        except Exception as e:
            #logger.error(f"Error updating Stripe Customer ID for {email}: {str(e)}")
            return False

    def Get_Stripe_Customer_ID(self, email):
        ''' Gets the Stripe Customer ID of the user '''
        try:
            # Convert email to lowercase for consistency
            email = email.lower()
            
            user = self.Get_User_Data(email)
            if user:
                return user.get('Stripe_Customer_ID')
            else:
                return None
        except Exception as e:
            #logger.error(f"Error getting Stripe Customer ID for {email}: {str(e)}")
            return None

    def Delete_Session(self, email):
        ''' Deletes the session of the user '''
        try:
            # Convert email to lowercase for consistency
            email = email.lower()
            
            result = Delete_Session_DB(email)
            return result
        except Exception as e:
            return False
        
    
    def Delete_Session_By_Date(self, email, date):
        ''' Deletes the session of the user for a specific date '''
        try:
            # Convert email to lowercase for consistency
            email = email.lower()
            
            result = Delete_Session_DB_By_Date(email, date)
            return result
        except Exception as e:
            return False

        # Fixed User class method
    def Get_Credits_History(self, email, page, limit):
        ''' Getting user credits history with pagination '''
        try:
            # Convert email to lowercase for consistency
            email = email.lower()
            
            result = Get_Credits_History_DB(email, page, limit)
            return result
        except Exception as e:
            return False




    def Delete_Session_By_Platform(self, email,date, platform):
        ''' Deletes the session of the user for a specific platform '''
        try:
            # Convert email to lowercase for consistency
            email = email.lower()
            
            result = Delete_Session_DB_By_Platform(email,date, platform)
            return result
        except Exception as e:
            return False
        
    def Delete_Session_By_Date_Platform_Time(self, email,date, platform, time):
        ''' Deletes the session of the user for a specific platform and time '''
        try:
            # Convert email to lowercase for consistency
            email = email.lower()
            
            result = Delete_Session_DB_By_Date_Platform_Time(email,date, platform, time)
            return result
        except Exception as e:
            return False
        
    def Get_Models(self, email):
        ''' Gets the models available to the user based on their API keys '''
        try:
            # Convert email to lowercase for consistency
            email = email.lower()
            
            user = self.Get_User_Data(email)
            if not user or 'API_Key' not in user:
                return []
            
            api_keys = user['API_Key']
            all_models = []
            
            # Get OpenAI models if API key exists
            if api_keys.get('OpenAI'):
                try:
                    openai_models = Get_Models_From_OpenAI(api_keys['OpenAI'])
                    if openai_models:
                        all_models.extend(openai_models)
                except Exception as e:
                    #logger.error(f"Error getting OpenAI models: {e}")
                    pass
            
            # Get Claude models if API key exists (Claude doesn't have a direct model listing API)
            if api_keys.get('Claude'):
                try:
                    # Add default Claude models
                    claude_models = ['claude-3-5-sonnet', 'claude-3-5-haiku']
                    all_models.extend(claude_models)
                except Exception as e:
                    #logger.error(f"Error adding Claude models: {e}")
                    pass
            
            # Get Gemini models if API key exists
            if api_keys.get('Gemini'):
                try:
                    gemini_models = Get_Models_From_Gemini(api_keys['Gemini'])
                    if gemini_models:
                        all_models.extend(gemini_models)
                except Exception as e:
                    #logger.error(f"Error getting Gemini models: {e}")
                    pass
            
            # Remove duplicates and return
            return list(set(all_models)) if all_models else []
            
        except Exception as e:
            #logger.error(f"Error getting models for user {email}: {e}")
            return []
        
    def Get_Platforms_By_Date(self, email,date):
        ''' Gets the platforms of the user for a specific date '''
        try:
            # Convert email to lowercase for consistency
            email = email.lower()
            
            result = Get_Platforms_By_Date_DB(email,date)
            return result
        except Exception as e:
            return False
        