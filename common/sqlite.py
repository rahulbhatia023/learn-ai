import sqlite3


def get_schema(sqlite_file):
    conn = sqlite3.connect(sqlite_file)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        schema = []

        def process_table(index):
            if index >= len(tables):
                conn.close()
                return "\n".join(schema)

            table_name, create_statement = tables[index]
            schema.append(f"Table: {table_name}")
            schema.append(f"CREATE statement: {create_statement}\n")

            cursor.execute(f"SELECT * FROM '{table_name}' LIMIT 3;")
            rows = cursor.fetchall()

            if rows:
                schema.append("Example rows:")
                for row in rows:
                    schema.append(str(row))
            schema.append("")

            return process_table(index + 1)

        return process_table(0)

    except sqlite3.Error as e:
        conn.close()
        raise Exception(f"Error: {str(e)}")


def execute_query(sqlite_file, query):
    conn = sqlite3.connect(sqlite_file)
    cursor = conn.cursor()

    try:
        cursor.execute(query)
        rows = cursor.fetchall()

        return [list(row) for row in rows]

    except sqlite3.Error as e:
        conn.close()
        raise Exception(f"Error: {str(e)}")
