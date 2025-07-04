from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import logging
import google.generativeai as genai
import json
import certifi

#logger = logging.get#logger(__name__)
load_dotenv()

uri = os.getenv('MONGO_URL')
client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
db = client["Translator-Data"]
collection = db["C1"]




def get_data_from_collections():
    """
    Retrieve Model_table and Packages data from MongoDB collections
    """
    try:
        # Setup MongoDB connection
        uri = os.getenv('MONGO_URL')
        if not uri:
            print("Error: MONGO_URL environment variable not found")
            return None, None
            
        client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())

        db = client["Translator-Data"]
        
        # Get collections
        model_cost_collection = db["Model Cost"]
        translation_packages_collection = db["Translation packages"]
        
        model_cost_doc = model_cost_collection.find_one({"_id": "model_costs"})
        
        packages_doc = translation_packages_collection.find_one({"_id": "pricing_packages"})
        
        client.close()
        
        return model_cost_doc['data'] if model_cost_doc else None, packages_doc['data'] if packages_doc else None
        
    except Exception as e:
        print(f"❌ Error retrieving data: {str(e)}")
        return None, None

def get_all_collections_info():
    """
    Get information about all collections in the database
    """
    try:
        uri = os.getenv('MONGO_URL')
        client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())

        db = client["Translator-Data"]
        
        collections = db.list_collection_names()
        for i, collection_name in enumerate(collections, 1):
            collection = db[collection_name]
            doc_count = collection.count_documents({})
        
        client.close()
        
    except Exception as e:
        print(f"❌ Error getting collection info: {str(e)}")

# Fixed Database function
def Get_Credits_History_DB(email, page, limit):
    ''' Getting user credits history with pagination '''
    email = email.lower()
    user = Find_User_DB(email)
    if user:
        # Get all transactions and sort by timestamp (newest first)
        all_transactions = user.get("CreditTransactions", [])
        all_transactions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Get the total number of transactions
        total_transactions = len(all_transactions)
        
        # Calculate the start and end indices for the current page
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        
        # Get the transactions for the current page
        page_transactions = all_transactions[start_idx:end_idx]
        
        return {
            'transactions': page_transactions,
            'total_count': total_transactions,
            'current_page': page,
            'limit': limit
        }
    else:
        return {
            'transactions': [],
            'total_count': 0,
            'current_page': page,
            'limit': limit
        }
    
    
def Get_Models_From_OpenAI(api_key):
    ''' Get available models from OpenAI '''
    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()

        # Filter models likely suitable for translation based on naming
        translation_keywords = ['translate', 'text', 'gpt', 'chat']
        filtered_models = [
            model.id for model in models.data
            if any(keyword in model.id.lower() for keyword in translation_keywords)
        ]

        # Remove models related to embeddings, TTS, images, etc.
        exclusion_keywords = ['embedding', 'tts', 'image', 'search', 'audio','transcribe']
        filtered_models = [
            model for model in filtered_models
            if not any(exclude in model.lower() for exclude in exclusion_keywords)
        ]

        return filtered_models
    except Exception as e:
        print("Error:", e)
        return False

def Get_Models_From_Gemini(api_key):
    ''' Get available models from Gemini '''
    try:
        genai.configure(api_key=api_key)
        
        # Get list of available models
        models = genai.list_models()
        
        # Filter models suitable for text generation/translation
        filtered_models = []
        for model in models:
            model_name = model.name.replace('models/', '')
            # Include models that support generateContent and are suitable for translation
            if hasattr(model, 'supported_generation_methods') and 'generateContent' in model.supported_generation_methods:
                filtered_models.append(model_name)
        
        return filtered_models if filtered_models else ['gemini-1.5-pro-latest', 'gemini-2.5-flash']
    except Exception as e:
        #logger.error(f"Error getting Gemini models: {e}")
        # Return default models if API call fails
        return ['gemini-1.5-pro-latest', 'gemini-2.5-flash', 'gemma-3-27b-it', 'learnlm-2.0-flash-experimental']

def check_openai_key(api_key):
    try:
        # Create OpenAI client with the provided API key
        client = OpenAI(api_key=api_key)
        
        # Make a basic request to list models
        models = client.models.list()
        # print("✅ API key is valid.")
        # print(f"✅ Found {len(list(models.data))} available models.")
        return True
    except Exception as e:
        error_str = str(e).lower()
        if "unauthorized" in error_str or "invalid api key" in error_str:
            # print("❌ Invalid API key.")
            return False
        else:
            # print(f"⚠️ An error occurred: {e}")
            return False

def check_gemini_key(api_key):
    try:
        # Configure Gemini with the provided API key
        genai.configure(api_key=api_key)
        
        # Make a basic request to list models
        models = list(genai.list_models())
        #logger.info(f"Gemini API key is valid. Found {len(models)} available models.")
        return True
    except Exception as e:
        error_str = str(e).lower()
        if "invalid api key" in error_str or "unauthorized" in error_str or "forbidden" in error_str:
            #logger.error("Invalid Gemini API key.")
            return False
        else:
            #logger.error(f"Error validating Gemini API key: {e}")
            return False

def Delete_Session_DB(email):
    """
    Delete all sessions of the user
    """
    email = email.lower()
    user = Find_User_DB(email)
    if user:
        # Unset the entire "Session" field
        collection.update_one({"Email": email}, {"$unset": {"Session": ""}})
        return True
    return False

def Delete_Session_DB_By_Date(email, date):
    """
    Delete the session of the user for a specific date
    """
    email = email.lower()
    user = Find_User_DB(email)
    if user:
        if date in user.get("Session", {}):
            path_to_delete = f"Session.{date}"
            collection.update_one({"Email": email}, {"$unset": {path_to_delete: ""}})
            
            # Check if Session object is empty after deletion and clean up
            updated_user = Find_User_DB(email)
            if updated_user and "Session" in updated_user and not updated_user["Session"]:
                collection.update_one({"Email": email}, {"$unset": {"Session": ""}})
            
            return True
        return False
    return False

def Delete_Session_DB_By_Platform(email, date, platform):
    """
    Delete the session of the user for a specific platform
    """
    email = email.lower()
    user = Find_User_DB(email)
    if user:
        if date in user.get("Session", {}) and platform in user["Session"][date]:
            path_to_delete = f"Session.{date}.{platform}"
            collection.update_one({"Email": email}, {"$unset": {path_to_delete: ""}})
            
            # Check if the date object is empty after platform deletion
            updated_user = Find_User_DB(email)
            if updated_user and "Session" in updated_user and date in updated_user["Session"]:
                if not updated_user["Session"][date]:  # Date object is empty
                    collection.update_one({"Email": email}, {"$unset": {f"Session.{date}": ""}})
                    
                    # Check if Session object is empty after date deletion
                    final_user = Find_User_DB(email)
                    if final_user and "Session" in final_user and not final_user["Session"]:
                        collection.update_one({"Email": email}, {"$unset": {"Session": ""}})
            
            return True
        return False
    return False

def Delete_Session_DB_By_Date_Platform_Time(email, date, platform, time):
    """
    Delete the session of the user for a specific platform and time
    """
    # Convert email to lowercase for consistency
    email = email.lower()
    user = Find_User_DB(email)
    if user:
        # Check if the path exists before trying to delete
        if (date in user["Session"] and 
            platform in user["Session"][date] and 
            time in user["Session"][date][platform]):

            # Build the dot-notation path for MongoDB unset
            path_to_delete = f"Session.{date}.{platform}.{time}"

            # Unset the key from the document
            collection.update_one({"Email": email}, {"$unset": {path_to_delete: ""}})

            # Check if platform object is empty after time deletion
            updated_user = Find_User_DB(email)
            if updated_user and "Session" in updated_user and date in updated_user["Session"]:
                if platform in updated_user["Session"][date]:
                    if not updated_user["Session"][date][platform]:  # Platform object is empty
                        collection.update_one({"Email": email}, {"$unset": {f"Session.{date}.{platform}": ""}})
                        
                        # Check if the date object is empty after platform deletion
                        second_updated_user = Find_User_DB(email)
                        if second_updated_user and "Session" in second_updated_user and date in second_updated_user["Session"]:
                            if not second_updated_user["Session"][date]:  # Date object is empty
                                collection.update_one({"Email": email}, {"$unset": {f"Session.{date}": ""}})
                                
                                # Check if Session object is empty after date deletion
                                final_user = Find_User_DB(email)
                                if final_user and "Session" in final_user and not final_user["Session"]:
                                    collection.update_one({"Email": email}, {"$unset": {"Session": ""}})

            return True
        else:
            return False  # Entry doesn't exist
    else:
        return False

def Get_Platforms_By_Date_DB(email,date):
    """
    Get the platforms of the user for a specific date
    """
    # Convert email to lowercase for consistency
    email = email.lower()
    user = Find_User_DB(email)
    if user:
        return user["Session"][date].keys()
    else:
        return False
    
def convert_objectid_to_str(obj):
    """
    Recursively convert ObjectId instances to strings in dictionaries and lists
    
    Args:
        obj: The object to convert (dict, list, or any other type)
        
    Returns:
        The object with ObjectId instances converted to strings
    """
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_objectid_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid_to_str(item) for item in obj]
    else:
        return obj

def Find_User_DB(email):
    """
    Find the user in the database

    Args:
        email: str

    Returns:
        dict: User data if found, None otherwise example return:
        {'_id': ObjectId('684a6553a5528cc6bb154da9'), 'Name': 'John Doe', 'Email': 'john.doe@example.com', 'Password': '1234567890', 'Created_At': '2025-06-12', 'API_Key': {'OpenAI': 'sk-proj-1234567890', 'Deepgram': '1234567890', 'Claude': '1234567890', 'Gemini' : '12312312312'}, 'Session': {'2022-01-01': {'Youtube': {'10:20:57': {'Original Text': 'Hello, how are you?', 'Translated Text': 'Bonjour, comment Ã§a va?', 'Summary': 'Bonjour, comment Ã§a va?', 'Original Language': 'French', 'Translated Language': 'English', 'Keywords': [{'keyword': 'speaker', 'sentiment': 0.0}, {'keyword': 'peur', 'sentiment': 0.0}, {'keyword': 'voir', 'sentiment': 0.0}, {'keyword': 'chantier', 'sentiment': 0.0}, {'keyword': 'rentre', 'sentiment': 0.0}]}, '10:21:00': {'Original Text': 'Hello, how are you?', 'Translated Text': 'Bonjour, comment Ã§a va?', 'Summary': 'Bonjour, comment Ã§a va?', 'Original Language': 'French', 'Translated Language': 'English', 'Keywords': [{'keyword': 'speaker', 'sentiment': 0.0}]}}, 'Google_Meet': {'10:20:57': {'Original Text': 'Hello, how are you?', 'Translated Text': 'Bonjour, comment Ã§a va?', 'Summary': 'Bonjour, comment Ã§a va?', 'Original Language': 'French', 'Translated Language': 'English', 'Keywords': [{'keyword': 'speaker', 'sentiment': 0.0}]}}}}}
    """
    # Convert email to lowercase for consistent searching
    email = email.lower()
    result = collection.find_one({"Email": email})
    
    # Convert any ObjectId instances to strings for JSON serialization
    if result:
        result = convert_objectid_to_str(result)
    
    return result

def get_User_Data(email):
    """
    Get the user data from the database
    """
    # Convert email to lowercase for consistency
    email = email.lower()
    user = Find_User_DB(email)
    # ObjectId conversion is already handled in Find_User_DB
    return user

def get_User_API_Key(email):
    """
    Get the user API key from the database
    """
    user = Find_User_DB(email)
    return user['API_Key']

def get_User_AI_Mode(email):
    """
    Get the user AI mode from the database
    """
    user = Find_User_DB(email)
    return user['Summary_and_Keywords']

def Authenticate_User(email, password):
    """
    Authenticate the user with the email and password

    Args:
        email: str
        password: str

    Returns:
        bool: True if authenticated, False otherwise
    """
    # Convert email to lowercase for consistency
    email = email.lower()
    user = Find_User_DB(email)
    # print(user)
    if user and user["Password"] == password:
        return True
    else:
        return False

def Create_User(Name, email, password, API_Key, Paid_User, Summary_and_Keywords, Credits):
    """
    Create a new user in the database

    Args:
        Name: str
        email: str
        password: str
        API_Key: dict ==> {"OpenAI": str, "Deepgram": str, "Claude": str, "Gemini": str}

    Returns:
        bool: True if created, False otherwise and if user already exists return False
    """
    # Convert email to lowercase for consistency
    email = email.lower()
    
    user = {
            "Name": Name,
            "Email": email,
            "Password": password,
            "Created_At": datetime.now().strftime("%Y-%m-%d"),
            "Paid_User": Paid_User,
            "Summary_and_Keywords": Summary_and_Keywords,
            "CreditTransactions": {
                
            },
            "Payment_History": {
                
            },
            "Credits": Credits,
            "API_Key": {
                "OpenAI": API_Key.get("OpenAI", ""),
                "Deepgram": API_Key.get("Deepgram", ""),
                "Claude": API_Key.get("Claude", ""),
                "Gemini": API_Key.get("Gemini", "")
            },
            "Session":{

            }
        }

    collection.insert_one(user)
    return True

def Add_Session(email, session,date):
    """
    Add a session to the user

    Args:
        email: str
        session: dict ==> {
                        "Original Text": str,
                        "Translation": str,
                        "summary": str,
                        "orginal language": str,
                        "translation language": str,
                    }

    Returns:
        bool: True if session added, False otherwise

    """
    # Convert email to lowercase for consistency
    email = email.lower()
    user = Find_User_DB(email)

    if user:
        user["Session"][date] = session
    else:
        return False

    collection.update_one({"Email": email}, {"$set": user})
    return True

def Get_Session(email):
    """
    Get the session of the user

    Args:
        email: str

    Returns:
        dict: Session data if found, None otherwise
    """
    # Convert email to lowercase for consistency
    email = email.lower()
    user = Find_User_DB(email)
    if user:
        return user["Session"]
    else:
        return False

def Save_Session(email, session, WebsiteName, date):
    """ 
    Find user in the Database then add session data into the data base
    The session will be added to that website at that data as there may be multiple sessions at that website on that date
    
    Args:
        email: str
        session: dict
        WebsiteName: str
    """
    # Convert email to lowercase for consistency
    email = email.lower()
    user = Find_User_DB(email)
    if user:
        user["Session"][date][WebsiteName].append(session)
    else:
        return False
    collection.update_one({"Email": email}, {"$set": user})
    return True

def Update_User_Keyword_and_Summary(email, Summary_and_Keywords):
    """
    Update the user's keyword and summary preference
    """
    try:
        # Convert email to lowercase for consistency
        email = email.lower()
        user = Find_User_DB(email)

        if user:
            # Use upsert to create the field if it doesn't exist
            collection.update_one(
                {"Email": email}, 
                {"$set": {"Summary_and_Keywords": Summary_and_Keywords}}, 
                upsert=True
            )
            return True
        else:
            return False
    except Exception as e:
        print(f"Error updating Summary and Keywords for {email}: {str(e)}")
        return False

def Add_User_To_DB(user):
    """ Adding user to the database """
    # Convert all keys to lowercase for consistency
    user = {key.lower(): value for key, value in user.items()}
    
    # Convert email to lowercase
    email = user.get("email", "").lower()
    
    if Find_User_DB(email):
        return False
    else:
        #logger.info(f"Adding user to the database: {user}")
        success = Create_User(
                Name=user.get("name", ""),
                email=user.get("email", ""),
                password=user.get("password", ""),
                API_Key=user.get("api_key", {}),
                Paid_User=user.get("paid_user", False),
                Summary_and_Keywords=user.get("summary_and_keywords", 1),
                Credits=user.get("Credits", 10)
            )
        #logger.info(f"User added to the database: {success}")
        return success