#!/usr/bin/env python3
"""
Improved API testing script for DL extraction service
"""
import requests
import json
import time
import sys
import os

def test_api(image_path: str, api_url: str):
    """Test the DL extraction API with detailed logging"""
    
    print(f"🚀 Testing DL Extraction API")
    print(f"📍 API URL: {api_url}")
    print(f"🖼️  Image: {image_path}")
    print("-" * 60)
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"❌ ERROR: Image file not found: {image_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
    print(f"📊 File size: {file_size:.2f} MB")
    
    if file_size > 16:
        print("⚠️  WARNING: File size exceeds 16MB limit")
    
    try:
        # Test health endpoint first
        print("\n🏥 Testing health endpoint...")
        try:
            health_response = requests.get(f"{api_url.rstrip('/extract-dl')}/health", timeout=10)
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"✅ Health check passed: {health_data}")
            else:
                print(f"⚠️  Health check returned: {health_response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Health check failed: {e}")
        
        # Prepare the request
        print(f"\n🔄 Sending request to {api_url}...")
        start_time = time.time()
        
        with open(image_path, 'rb') as image_file:
            files = {'dl_image': image_file}
            
            # Make the request with extended timeout for first request (model loading)
            response = requests.post(
                api_url, 
                files=files, 
                timeout=900  # 15 minutes timeout for first request
            )
        
        end_time = time.time()
        request_time = end_time - start_time
        
        print(f"⏱️  Request completed in {request_time:.2f} seconds")
        print(f"📡 Response status: {response.status_code}")
        print(f"📄 Response headers: {dict(response.headers)}")
        
        # Handle the response
        if response.status_code == 200:
            try:
                result = response.json()
                print("✅ SUCCESS!")
                print(f"🆔 DL Number: {result.get('dl_number', 'Not found')}")
                
                if 'all_dl_numbers' in result:
                    print(f"📋 All DL Numbers: {result['all_dl_numbers']}")
                
                if 'processing_time' in result:
                    print(f"⚡ Server processing time: {result['processing_time']:.2f}s")
                
                if 'raw_text' in result:
                    print(f"📝 Raw OCR text (preview): {result['raw_text'][:200]}...")
                
                return True
                
            except json.JSONDecodeError:
                print(f"✅ SUCCESS (Plain text response): {response.text}")
                return True
                
        elif response.status_code == 404:
            try:
                error_data = response.json()
                print("⚠️  NO DL NUMBER FOUND")
                print(f"📝 Raw text: {error_data.get('raw_text', 'Not available')}")
                if 'processing_time' in error_data:
                    print(f"⚡ Processing time: {error_data['processing_time']:.2f}s")
            except json.JSONDecodeError:
                print(f"⚠️  NO DL NUMBER FOUND: {response.text}")
            return False
            
        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"📝 Error details: {error_data}")
            except json.JSONDecodeError:
                print(f"📝 Raw error response: {response.text}")
            return False
    
    except requests.exceptions.Timeout:
        print("⏰ ERROR: Request timed out")
        print("💡 This might be normal for the first request (model loading)")
        return False
        
    except requests.exceptions.ConnectionError:
        print("🔌 ERROR: Could not connect to the API")
        print("💡 Check if the URL is correct and the service is running")
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"🌐 ERROR: Request failed: {e}")
        return False
        
    except Exception as e:
        print(f"💥 UNEXPECTED ERROR: {e}")
        return False

def main():
    """Main function to run the API test"""
    
    # Default values - modify these for your setup
    default_api_url = "https://dl-extractor-YOUR-HASH-uc.a.run.app/extract-dl"
    default_image_path = "test_dl.jpg"
    
    # Get parameters from command line or use defaults
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    else:
        image_path = input(f"Enter image path (default: {default_image_path}): ").strip()
        if not image_path:
            image_path = default_image_path
    
    if len(sys.argv) >= 3:
        api_url = sys.argv[2]
    else:
        api_url = input(f"Enter API URL (default: {default_api_url}): ").strip()
        if not api_url:
            api_url = default_api_url
    
    print("=" * 80)
    print("🔬 DL EXTRACTION API TEST")
    print("=" * 80)
    
    # Run the test
    success = test_api(image_path, api_url)
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 TEST COMPLETED SUCCESSFULLY!")
    else:
        print("❌ TEST FAILED - Check the logs above for details")
    print("=" * 80)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)