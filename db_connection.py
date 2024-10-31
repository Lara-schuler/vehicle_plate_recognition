import mysql.connector
from mysql.connector import Error

def create_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',  # Alterado para 'localhost', ajuste conforme necessário
            user='root',
            password='pacoquinha',
            database='vehicle_recognition'
        )
        
        if connection.is_connected():
            print("Conexão com o banco de dados estabelecida com sucesso!")
        
        return connection

    except Error as e:
        print(f"Erro ao conectar ao banco de dados: {e}")
        return None
