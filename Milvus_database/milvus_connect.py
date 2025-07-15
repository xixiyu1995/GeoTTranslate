from pymilvus import exceptions
from pymilvus import MilvusClient


class Milvus_Connect():
    def __init__(self):
        print("Milvus Init...")
        try:
            self.milvus_client = MilvusClient(uri="http://127.0.0.1:19530", user="yyy", password="123456")
            # 检查健康状态
            self.check_connection()

        except exceptions.MilvusException as e:
            print(f"Failed to connect to Milvus: {e}")

    def check_connection(self):
        try:
            health_status = self.milvus_client
            if health_status:
                print(f"Milvus connection is successful, server version: {health_status}")

        except exceptions.MilvusException as e:
            print(f"Connection check failed: {e}")


# 使用示例
if __name__ == "__main__":
    milvus_connection = Milvus_Connect()


