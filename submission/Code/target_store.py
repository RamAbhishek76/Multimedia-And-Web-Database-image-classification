from database_connection import connect_to_mongo

client=connect_to_mongo()
db=client.cse515
collection_with_target=db.Phase2
collection_needing_target=db.phase2_ls1

merged_collection = db.merged_collection

for doc in collection_needing_target.find():
    # Fetch the extra field from the extra_collection based on a shared identifier (e.g., user_id)
    extra_data = collection_with_target.find_one({"image_id": doc["image_id"]})
    
    if extra_data:
        doc["target"] = extra_data.get("target", None)  
        
    merged_collection.insert_one(doc)
    
assert collection_needing_target.count_documents({}) <= merged_collection.count_documents({}), "Data merge/migration might have issues!"

client.close()