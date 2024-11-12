from db_connection import get_connection

def save_vehicle_data(plate):
    print(f"Tentando salvar a placa: {plate}")
    connection = get_connection()
    
    if connection is None:
        print("Erro: Falha ao obter conexão do pool de conexões.")
        return

    cursor = connection.cursor()
    try:
        # Query de inserção de dados
        query = "INSERT INTO plate_data (plate) VALUES (%s)"
        cursor.execute(query, (plate,))
        connection.commit()

        # Confirmação da inserção
        cursor.execute("SELECT plate FROM plate_data WHERE plate = %s LIMIT 1", (plate,))
        result = cursor.fetchone()  # Usa fetchone() para obter um único registro
        if result:
            print(f"Confirmação: Placa {result[0]} foi encontrada no banco de dados após a inserção.")
        else:
            print(f"Falha: Placa {plate} não foi encontrada no banco de dados após a inserção.")

    except Exception as e:
        print(f"Erro ao salvar a placa {plate} no banco de dados: {e}")

    finally:
        cursor.close()
        connection.close()  # Conexão é devolvida ao pool
