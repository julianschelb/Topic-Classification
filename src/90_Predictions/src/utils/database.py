# ===========================================================================
#                            Database Operation Helpers
# ===========================================================================

from typing import List
from dotenv import load_dotenv
from bson import ObjectId
import pymongo as pm
import gridfs
import os

# --------------------------------- Connection --------------------------------


def getConnection(
    connection_string: str = "", database_name: str = "", use_dotenv: bool = False
):
    "Returns MongoDB and GridFS connection"

    # Load config from config file
    if use_dotenv:
        load_dotenv()
        connection_string = os.getenv("CONNECTION_STRING")
        database_name = os.getenv("DATABASE_NAME")

    # Use connection string
    conn = pm.MongoClient(connection_string)
    db = conn[database_name]
    fs = gridfs.GridFS(db)

    return fs, db


# --------------------------------- Documents --------------------------------


def getLatestBatchID(db) -> int:
    "Returns the highest existing batch ID"
    result = db.pages.content.find_one(sort=[("batch_id", pm.DESCENDING)])
    latest_batch = result.get("batch_id", 0) if result is not None else 0
    return latest_batch


def getFirstBatchID(db) -> int:
    "Returns the lowest existing batch ID with unprocessed pages"

    result = db.pages.content.find_one(
        filter={"status": "UNPROCESSED"}, sort=[("batch_id", pm.ASCENDING)]
    )
    batch_id = result.get("batch_id", 0) if result is not None else 0
    return batch_id


def updateTask(db, id: str, values: dict = {}):
    "Updates scraping task in database"
    pass
    # filter = {"_id": ObjectId(id)}
    # values = {
    #     "$set": {**values},
    #     "$inc": {"tries": 1},
    # }
    # r = db.pages.content.update_one(filter, values)
    # return r


def fetchTasks(
    db,
    batch_id: int,
    status: str,
    http_series: List[str],
    limit: int = 0,
    skip: int = 0,
    fields: dict = {},
):
    """Returns a batch of scraping tasks"""

    # Add status code to fields
    fields["status_code"] = 1
    query = {"$and": []}

    if batch_id and status:
        query["$and"] = [{"status": status}, {"batch_id": batch_id}]
    elif status:
        # Consider all batches if no batch ID specified
        query["$and"] = [{"status": status}]
    elif batch_id:
        # Consider all batches if no batch ID specified
        query["$and"] = [{"batch_id": batch_id}]

    # Filtering out http response status codes
    filtered_status_codes = []
    for code in http_series:
        # prepare error message if code is invalid
        valueError = ValueError(f"Invalid HTTP response status code: {code}")

        # Check if code is valid
        if len(code) != 3:  # 3 digits
            raise valueError

        if code == "xxx":
            # don't filter for status codes
            filtered_status_codes = []
            break

        # only digits or x's
        for character in code:
            if character != "x" and not character.isdigit():
                raise valueError

        # replace x with \d for regex
        code = code.replace("x", "\d")

        # Add status code to filtered list
        filtered_status_codes.append(code)

    # Add status code filter to query
    if len(filtered_status_codes) > 0:
        regex = f"({'|'.join(filtered_status_codes)})"

        query["$and"].append(
            {"$expr":
                {"$regexMatch":
                    {
                        "input": {"$toString": "$status_code"},
                        "regex": regex
                    }
                 }
             })

    # Sorting requires a lot of memory
    tasks = db.pages.content.find(query, fields).limit(limit).skip(skip)

    return list(tasks)


def fetchTasksAllBatches(
    db, status: str, http_series: List[str], limit: int = 0, skip: int = 0, fields: dict = {}
):
    """Returns scraping tasks across all batches"""
    return fetchTasks(db, None, status, http_series, limit, skip, fields)


def insertContent(db, content: dict):
    """Inserts content into the database"""
    filter_condition = {"_id": content["_id"]}
    r = db.pages.content.extracted.update_one(
        filter_condition, {"$set": content}, upsert=True)
    return r


def fetchTasksContent(
    db,
    limit: int = 0,
    skip: int = 0,
    query={},
    fields: dict = {},
):

    # Sorting requires a lot of memory
    tasks = db.pages.content.extracted.find(
        query, fields).limit(limit).skip(skip)
    return list(tasks)

# --------------------------------- Files --------------------------------


def getPageContent(fs: gridfs, id: str, encoding="UTF-8"):
    """Retrieves a file from GridFS"""
    f = fs.get(ObjectId(id))
    return f.read().decode(encoding)


def getPageContentInfo(db, id: str):
    """Retrieves a file from GridFS"""
    info = db.fs.files.find_one({"_id": ObjectId(id)})
    return dict(info)


def savePageContent(fs, content, encoding="UTF-8", attr={}):
    """Saves a file in GridFS"""
    if content and len(content) > 0:
        if type(content) == str:
            content = content.encode(encoding)
        file_id = fs.put(content, **attr)
        return file_id
    # else:
    #    raise ValueError("File must not be emtpy")
    return None


# if __name__ == "__main__":

#     fs, db = getConnection(use_dotenv=True)
#     print(getLatestBatchID(db))
#     print(getFirstBatchID(db))

#     meta = {"url": "www.example.com"}
#     id = savePageContent(fs, "<h1>This is a Test</h1>", attr=meta)
#     print(id)

#     f = getPageContent(fs, id)
#     print(f)

#     f = getPageContentInfo(db, id)
#     print(f)
