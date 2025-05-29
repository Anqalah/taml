import mysql.connector

def database_conn():
    try:
        conn = mysql.connector.connect(
            host="caboose.proxy.rlwy.net",
            port=26276,
            user="root",
            password="cMtooYtfEThsDlsYhmHliOajEanVLXyP",
            database="railway"
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None