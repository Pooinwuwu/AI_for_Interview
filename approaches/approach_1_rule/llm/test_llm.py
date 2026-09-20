"""
test_env.py — ตรวจสอบว่าอ่าน API key ได้
"""
from dotenv import load_dotenv
load_dotenv()

import os
import sys

key = os.environ.get("GEMINI_API_KEY")

print("=" * 50)
if not key:
    print("❌ ไม่พบ GEMINI_API_KEY")
    print("\nวิธีแก้:")
    print("  1. สร้างไฟล์ .env ที่ root โปรเจกต์")
    print("  2. ใส่บรรทัด: GEMINI_API_KEY=AIzaSy...")
    print("  3. รันใหม่")
    sys.exit(1)

print(f"✅ พบ GEMINI_API_KEY")
print(f"   ความยาว: {len(key)} ตัวอักษร")
print(f"   ขึ้นต้น: {key[:8]}...")
print(f"   ลงท้าย: ...{key[-4:]}")

# ทดสอบเรียก Gemini
try:
    from google import genai as genai
    client = genai.Client(api_key=key)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents="Hello, Gemini! How are you today?",

    )
    print(f"\n✅ เรียก Gemini สำเร็จ: {response.text}")


except Exception as e:
    print(f"\n❌ เรียก Gemini ไม่สำเร็จ: {e}")
print("=" * 50)