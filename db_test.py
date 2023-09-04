from pymongo import MongoClient

try:
    # uri = "mongodb+srv://abhinav:Abhinav123@cluster0.ogwzfld.mongodb.net/?retryWrites=true&w=majority"
    uri = "mongodb://localhost:27017/cse515_project_phase1"
    client = MongoClient(uri)
    print("Connected successfully!!!")
    # # database
    # db = client.database

    # # Created or Switched to collection names: my_gfg_collection
    # collection = db.caltect_101_features

    # cursor = collection.find()
    # for record in cursor:
    #     print(record)
except e:
	print(e)


