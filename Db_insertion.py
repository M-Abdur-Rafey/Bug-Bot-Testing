Model_table = {
    "OpenAI": {
      "gpt-4.1": 10.0,
      "gpt-4.1-mini": 2.0,
      "gpt-4.1-nano": 0.5,
      "gpt-4o": 12.50,
      "gpt-4o-mini": 1.00,
    },
    "Claude": {
        "claude-3-5-sonnet": 0.8,
        "claude-3-5-haiku": 0.5,
    },
    "Gemini": {
        "gemini-1.5-pro-latest": 0.6,
        "gemini-2.5-flash": 0.4,
    }
}

# Packages = {
#     "Basic": {
#         "Credits": 1000,
#         "Price": 10,
#         "Description": "Perfect for light usage",
#         "Features": [
#             "1,000 Credits",
#             "No expiration",
#             "All models included",
#             "Email support"
#         ],
#         "Link": "https://buy.stripe.com/test_cNi7sEdVc8g8aV52xSg3600",
#         "Popular": False
#     },
#     "Pro": {
#         "Credits": 5000,
#         "Price": 40,
#         "Description": "Most popular choice",
#         "Features": [
#             "5,000 Credits",
#             "No expiration", 
#             "All models included",
#             "Priority support",
#             "20% savings"
#         ],
#         "Link": "https://buy.stripe.com/test_7sY14g4kCcwo7ITa0kg3601",
#         "Popular": True,
#         "Discount": "20% off"
#     },
#     "Enterprise": {
#         "Credits": 10000,
#         "Price": 70,
#         "Description": "Best value for heavy users",
#         "Features": [
#             "10,000 Credits",
#             "No expiration",
#             "All models included",
#             "Priority support",
#             "Bulk discount",
#             "30% savings"
#         ],
#         "Link": "https://buy.stripe.com/test_8x23co7wO3ZS1kv5K4g3602",
#         "Popular": False,
#         "Discount": "30% off"
#     }
# }

# Database insertion code
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def insert_data_to_collections():
    """
    Insert Model_table and Packages data into MongoDB collections
    """
    try:
        # Setup MongoDB connection
        uri = os.getenv('MONGO_URL')
        if not uri:
            print("Error: MONGO_URL environment variable not found")
            return False
            
        client = MongoClient(uri, server_api=ServerApi('1'))
        db = client["Translator-Data"]
        
        # Get collections
        model_cost_collection = db["Model Cost"]
        translation_packages_collection = db["Translation packages"]
        
        # Clear existing data (optional - remove if you want to preserve existing data)
        print("Clearing existing data...")
        model_cost_collection.delete_many({})
        translation_packages_collection.delete_many({})
        
        # Insert Model_table into "Model Cost" collection
        print("Inserting Model_table into 'Model Cost' collection...")
        model_cost_collection.insert_one({
            "_id": "model_costs",
            "data": Model_table,
            "inserted_at": "2024-01-20",
            "description": "Model pricing information for different AI providers"
        })
        
        # Insert Packages into "Translation packages" collection  
        print("Inserting Packages into 'Translation packages' collection...")
        translation_packages_collection.insert_one({
            "_id": "pricing_packages", 
            "data": Packages,
            "inserted_at": "2024-01-20",
            "description": "Available credit packages for users"
        })
        
        print("✅ Successfully inserted data into MongoDB collections!")
        print(f"- Model Cost collection: {model_cost_collection.count_documents({})} documents")
        print(f"- Translation packages collection: {translation_packages_collection.count_documents({})} documents")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Error inserting data: {str(e)}")
        return False

if __name__ == "__main__":
    insert_data_to_collections()
