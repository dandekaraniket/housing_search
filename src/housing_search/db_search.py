import sqlite3

class DBHelper:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = None

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.db_name)
            return True
        except sqlite3.Error as e:
            print(f"Connection error: {e}")
            return False

    def close(self):
        if self.conn:
            self.conn.close()

    def create_table(self, table_name, schema):
        try:
            cursor = self.conn.cursor()
            query = f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})"
            cursor.execute(query)
            self.conn.commit()
            print(f"Table '{table_name}' created successfully.")
            return True
        except sqlite3.Error as e:
            print(f"Error creating table: {e}")
            self.conn.rollback()  
            return False

    def delete_table(self, table_name):
        try:
            cursor = self.conn.cursor()
            query = f"DROP TABLE IF EXISTS {table_name}"
            cursor.execute(query)
            self.conn.commit()
            print(f"Table '{table_name}' deleted successfully.")
            return True
        except sqlite3.Error as e:
            print(f"Error deleting table: {e}")
            self.conn.rollback()
            return False

    def insert_data(self, table_name, data):
        try:
            cursor = self.conn.cursor()
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?"] * len(data))
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            cursor.execute(query, tuple(data.values())) 
            self.conn.commit()
            print("Data inserted successfully.")
            return True
        except sqlite3.Error as e:
            print(f"Error inserting data: {e}")
            self.conn.rollback()
            return False

    def delete_data(self, table_name, where_clause=""):
      try:
          cursor = self.conn.cursor()
          query = f"DELETE FROM {table_name}"
          if where_clause:
              query += f" WHERE {where_clause}"
          cursor.execute(query)
          self.conn.commit()
          print("Data deleted successfully.")
          return True
      except sqlite3.Error as e:
          print(f"Error deleting data: {e}")
          self.conn.rollback()
          return False
      
    def query_data_specific(self, table_name, key):
        try:
            cursor = self.conn.cursor()
            get_query = f'SELECT text, document_id FROM {table_name} WHERE embedding = {key}'
            query_data = cursor.execute(get_query)
            query_result = query_data.fetchone() 
            self.conn.commit()
            return query_result
        except sqlite3.Error as e:
            print(f"Error inserting data: {e}")
            self.conn.rollback()
            return False
    
    def query_data_document_table(self, table_name, key):
        try:
            cursor = self.conn.cursor()
            get_query = f'SELECT filename FROM {table_name} WHERE id = {key}'
            query_data = cursor.execute(get_query)
            query_result = query_data.fetchone() 
            self.conn.commit()
            return query_result
        except sqlite3.Error as e:
            print(f"Error inserting data: {e}")
            self.conn.rollback()
            return False

    def update_table(self, table, column, column_value, update_paramter, update_value):
        try:
            cursor = self.conn.cursor()
            update_query = f"UPDATE {table} SET {column} = {column_value} WHERE {update_paramter} = \"{update_value}\""
            cursor.execute(update_query)
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error updating chunk: {e}")
            self.conn.rollback()
            return False

    def query_data(self, table_name, embedding_key=None, doc_id_key=None):
        try:
            cursor = self.conn.cursor()
            query = f'SELECT * FROM {table_name}'  

            conditions = []
            params = []

            if embedding_key is not None:
                conditions.append('embedding = ?')
                params.append(embedding_key)

            if doc_id_key is not None:
                conditions.append('document_id = ?')
                params.append(doc_id_key)

            if conditions:
                query += ' WHERE ' + ' AND '.join(conditions)

            cursor.execute(query, tuple(params)) 
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            result = [dict(zip(columns, row)) for row in rows] 

            if result:
                return result
            else:
                return None 

        except sqlite3.Error as e:
            print(f"Error querying data: {e}")
            return None
        except TypeError:
            print("Embedding key must be int or str, doc id key must be int.")
            return None

    def query_document_splitter_table(self, table_name):
        try:
            cursor = self.conn.cursor()
            get_query = f'SELECT document_name FROM {table_name} where Chunked = 0'
            #added a condition to return those documents where Chunked=0 meaning they are not chunked yet&&&
            query_data = cursor.execute(get_query)
            query_result = query_data.fetchall()
            self.conn.commit()
            return query_result
        except sqlite3.Error as e:
            print(f"Error inserting data: {e}")
            self.conn.rollback()
            return False
    
    def query_document_name_from_splitter_table(self, table_name,docname):
        try:
            cursor = self.conn.cursor()
            get_query = f'SELECT document_name FROM {table_name} where document_name LIKE \'%{docname}%\''
            #added a condition to return those documents where Chunked=0 meaning they are not chunked yet&&&
            query_data = cursor.execute(get_query)
            query_result = query_data.fetchall()
            self.conn.commit()
            return query_result
        except sqlite3.Error as e:
            print(f"Error inserting data: {e}")
            self.conn.rollback()
            return False
        
    def query_document_splitter_table_splits(self, table_name, document_name):
        try:
            cursor = self.conn.cursor()
            document_name = '"' + document_name + '"'
            get_query = f'SELECT document_splits FROM {table_name} where document_name={document_name}'
            query_data = cursor.execute(get_query)
            query_result = query_data.fetchall()
            self.conn.commit()
            return query_result
        except sqlite3.Error as e:
            print(f"Error inserting data: {e}")
            self.conn.rollback()
            return False
        
    
    