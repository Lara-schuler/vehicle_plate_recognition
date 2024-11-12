import mysql.connector
from mysql.connector import pooling, Error

# Criando um pool de conexões
connection_pool = pooling.MySQLConnectionPool(
    pool_name="vehicle_recognition_pool",
    pool_size=5,  # Tamanho do pool pode ser ajustado conforme a necessidade
    host='localhost',
    user='root',
    password='pacoquinha',
    database='vehicle_recognition'
)

def get_connection():
    try:
        connection = connection_pool.get_connection()
        if connection.is_connected():
            print("Conexão obtida do pool com sucesso!")
        return connection

    except Error as e:
        print(f"Erro ao obter conexão do pool: {e}")
        return None
