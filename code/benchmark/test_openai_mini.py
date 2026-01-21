#!/usr/bin/env python3
"""Test script to inspect gpt-5-mini API response structure."""

from openai import OpenAI
import json

client = OpenAI()

# Simple test with text only
print("=" * 60)
print("Test 1: Text-only request")
print("=" * 60)

resp = client.responses.create(
    model="gpt-5.2",
    input=[{"role": "user", "content": [{"type": "input_text", "text": "Say yes or no: Is the sky blue?"}]}],
    max_output_tokens=320,
)

print(f"Response type: {type(resp)}")
print(f"Response attributes: {[x for x in dir(resp) if not x.startswith('_')]}")
print(f"\nResponse object:")
print(f"  str(resp): {str(resp)}")
print(f"  repr(resp): {repr(resp)}")

# Try to access common attributes
print("\nChecking common attributes:")
for attr in ["output", "output_text", "text", "content", "choices", "response", "data"]:
    if hasattr(resp, attr):
        val = getattr(resp, attr)
        print(f"  {attr}: {repr(val)} (type: {type(val)})")
        if isinstance(val, (list, dict)):
            print(f"    JSON: {json.dumps(val, indent=2, default=str)}")

# Try to convert to dict if possible
if hasattr(resp, "model_dump"):
    print(f"\nmodel_dump(): {resp.model_dump()}")
if hasattr(resp, "dict"):
    print(f"\ndict(): {resp.dict()}")
if hasattr(resp, "__dict__"):
    print(f"\n__dict__: {resp.__dict__}")

print("\n" + "=" * 60)
print("Test 2: With image (single frame)")
print("=" * 60)

# Test with an image
from PIL import Image
import io
import base64

# Create a simple test image
img = Image.new('RGB', (100, 100), color='red')
buf = io.BytesIO()
img.save(buf, format='JPEG')
b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
data_url = f"data:image/jpeg;base64,{b64}"

resp2 = client.responses.create(
    model="gpt-5-mini",
    input=[{"role": "user", "content": [
        {"type": "input_text", "text": "What color is this image? Answer yes or no."},
        {"type": "input_image", "image_url": data_url}
    ]}],
    max_output_tokens=320,
)

print(f"Response type: {type(resp2)}")
print(f"Response attributes: {[x for x in dir(resp2) if not x.startswith('_')]}")

for attr in ["output", "output_text", "text", "content", "choices", "response", "data"]:
    if hasattr(resp2, attr):
        val = getattr(resp2, attr)
        print(f"  {attr}: {repr(val)} (type: {type(val)})")
        if isinstance(val, (list, dict)):
            print(f"    JSON: {json.dumps(val, indent=2, default=str)}")

