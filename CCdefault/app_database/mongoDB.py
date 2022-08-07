import os
import sys
import pymongo

from CCdefault.app_logger import App_Logger
from CCdefault.app_exception.exception import App_Exception




CONNECTION_STRING = os.getenv('MONGODB_CONNSTRING')
DATABASE_NAME = "Credit"

lg = App_Logger("Database_operations")


class MongoDB:
    """class for mongo db operations"""

    def __init__(self, collection_name, drop_collection=False):
        """Initialize the class with the database name and collection name
        the class initialization the class with the below argument 
        Args:
            collection_name : collection name
        """

        try:
            conn = pymongo.MongoClient(CONNECTION_STRING)
            lg.debug('connection to mongo db successful')
            self.__db = conn[DATABASE_NAME]
            if drop_collection:
                self.Drop_Collection(collection_name)
                lg.debug(f'drop collection {collection_name}from mongo db successful')
            self.__collection = self.__db[collection_name]

        except Exception as e:
            lg.error('error in get connection to mongo db %s', e)
            raise App_Exception(e, sys) from e
        lg.debug('connection to mongo db successful')

    def checkexistence_col(self, COLLECTION_NAME):

        """It verifies the existence of collection name
        Collection_NAME: collection name
        returns True if collection exists else False"""

        collection_list = self.__db.list_collection_names()

        if COLLECTION_NAME in collection_list:
            lg.debug(f"Collection:'{COLLECTION_NAME}' in Database:'' exists")
            return True

        lg.error(f"Collection:'{COLLECTION_NAME}' in Database:' does not exists OR \n\
        no documents are present in the collection")
        return False

    def Insert_One(self, data):
        """insert one data into mongo dd
        Args:
            data (formatted ): data to be inserted into mongo db
            
            {Key : Value}
            
        Returns:
            True if insertion is successful else False
        """
        try:
            self.__collection.insert_one(data)
        except Exception as e:
            lg.debug('error in insert data into mongo db %s', e)
            raise App_Exception(e, sys) from e
        lg.debug('insert data into mongo db successful%s')
        return True

    def Insert_Many(self, data):
        """insert many data into mongo dd
        Args:
            data (formatted ): data to be inserted into mongo db
            
            {Key : Value}
            
        Returns:
            True if insertion is successful else False
        """
        lg.debug('insert many data into mongo db')
        try:
            self.__collection.insert_many(data)
        except Exception as e:
            lg.critical('error in insert many data into mongo db %s', e)

            raise App_Exception(e, sys) from e
        lg.debug('insert many data into mongo db successful')
        return True

    def Find_One(self, query={}):
        """find one data from mongo db
        if query is not provided then it will return the first document
        """

        lg.debug('find one data from mongo db')
        try:
            return self.__collection.find_one(query)
        except Exception as e:
            lg.critical('error in find one data from mongo db %s', e)
            raise App_Exception(e, sys) from e

    def Find_Many(self, query={}, limit=2000):
        """find many data from mongo db
        if query is not provided then it will return all the documents
        """

        try:
            lg.debug('find many data from mongo db')
            return self.__collection.find(query).limit(limit)
        except Exception as e:
            lg.critical('error in find many data from mongo db %s', e)
            return False

    def Drop_Collection(self, collection):
        """drop collection from mongo db
        Args:
            collection: collection name to be dropped
           
        Returns:
            True if drop is successful else False"""

        if self.checkexistence_col(collection):
            lg.debug('drop collection found in DB')
            try:
                lg.debug(f'drop collection{collection}from mongo db')
                self.__collection = self.__db[collection]
                self.__collection.drop()
            except Exception as e:
                lg.critical('error in drop collection from mongo db %s', e)
                raise App_Exception(e, sys) from e
            lg.debug('drop collection from mongo db successful')
            return True
        else:
            lg.error('collection not present in the database')
            return 'collection not present in the database'

    if __name__ == '__main__':
        pass