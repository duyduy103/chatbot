import os
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

print("File .env:", ENV_FILE)
print("File tồn tại:", ENV_FILE.exists())

load_dotenv(ENV_FILE, override=True)

# Kiểm tra tên biến thường gặp, không in giá trị bí mật
for name in ("MONGODB_URI", "MONGO_URI", "MONGODB_CONNECTION_STRING"):
    print(f"{name}:", "Có" if os.getenv(name) else "Không có")

uri = (
    os.getenv("MONGODB_URI")
    or os.getenv("MONGO_URI")
    or os.getenv("MONGODB_CONNECTION_STRING")
)

if not uri:
    print("Không tìm thấy URI. Hãy kiểm tra tên biến trong .env.")
else:
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=10000)
        print(client.admin.command("ping"))
        print("MongoDB connection OK")
    except Exception as e:
        print(type(e).__name__, str(e))
    finally:
        if "client" in locals():
            client.close()