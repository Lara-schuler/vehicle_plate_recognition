from db_connection import create_connection

def save_vehicle_data(plate):
    connection = create_connection()
    cursor = connection.cursor()

    # Inserir os dados da placa e data/hora no banco de dados
    query = "INSERT INTO plate_data (plate) VALUES (%s)"
    cursor.execute(query, (plate,))

    connection.commit()
    cursor.close()
    connection.close()
