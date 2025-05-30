import mysql.connector

password = '1234'
instance = 'police'


conn = mysql.connector.connect(
        host='localhost',
        user='root',
        passwd=password,
        database=instance
)
cursor = conn.cursor(buffered=True)
tables = ['stolenlicenseplates', 'videolicenseplates']
query = 'DELETE FROM %s'

for table in tables:

    try:

        formatted_query = query % table
        cursor.execute(formatted_query)
        conn.commit()

        print(f'Table {table} successfully deleted!')

    except mysql.connector.Error as err:

        conn.rollback()
        print(f"Error: {err}")