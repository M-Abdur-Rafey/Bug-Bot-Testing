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

from Autentication import get_data_from_collections

Model_table, Packages = get_data_from_collections()
