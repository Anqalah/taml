from flask import Flask, jsonify
import mysql.connector
import config 

class Database(object):
    @staticmethod
    def get_users(id_player):
        conn = None
        try:
            conn = config.database_conn()
            if conn is None:
                return jsonify({"error": "Could not connect to database"}), 500

            cursor = conn.cursor(dictionary=True)
            cursor.execute('SELECT face_image FROM students WHERE id = %s', (id_player,))
            results = cursor.fetchall()
            return results

        except Exception as e:
            return None, str(e)
        finally:
            if conn and conn.is_connected():
                conn.close()
