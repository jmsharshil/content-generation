import os
import io
import base64
import uuid
import requests
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from openai import AzureOpenAI
try:
    from PIL import Image
except ImportError:
    pass

def generate_post_content(prompt: str) -> str:
    endpoint_url = os.environ.get("ENDPOINT_URL", "").strip()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Check if this is an Azure OpenAI key
    if endpoint_url and 'azure.com' in endpoint_url:
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2023-12-01-preview",
            azure_endpoint=endpoint_url
        )
        # Note: In Azure, 'model' must match your exact deployment name.
        model_name = "gpt-4o-mini"
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                # {"role": "system", "content": (
                #     "You are a top-tier expert social media manager, creative director, and copywriter. "
                #     "Your goal is to write highly engaging, human-like social media posts that convert and drive engagement. "
                #     "You will be provided with information from a brand setup and a PDF document. You MUST use all the relevant details and context provided in these documents to generate the content. "
                #     "When asked to generate multiple options, you must ensure they are highly diverse: use different hooks (questions, statements, statistics), vary the pacing, and try different emotional angles. "
                #     "CRITICAL: Do NOT use any markdown formatting. Do NOT use bullet points, hyphens (-), or dashes for lists. Write in natural flowing paragraphs. "
                #     "Do NOT sound like an AI. Completely avoid AI buzzwords like 'Unlock', 'Dive in', 'In today\\'s digital landscape', 'Elevate', or 'Discover'. Keep the tone conversational, authentic, and natural. "
                #     "STRICT RULE: If generating multiple options, you MUST separate each option strictly with the string '---OPTION---' on a new line. Do NOT output labels like 'Option 1' or 'Variation 2'. "
                #     "STRICT RULE: Output ONLY the raw post content. NEVER include conversational filler like 'Here are your posts', 'Sure', or 'You\\'re welcome!'. Start the first post immediately."
                # )},
                {"role": "system", "content": (
                    "You are a top-tier Direct-Response Copywriter and Social Media Operations Director. "
                    "Your goal is to write highly engaging, human-like social media posts that convert, drive engagement, and are engineered for social media feeds. "
                    "You will be provided with information from a brand setup and a PDF document. You MUST use all the relevant details and context provided in these documents to generate the content. "
                    "When asked to generate multiple options, you must ensure they are highly diverse by using pattern-interrupt hooks (e.g., 3-second curiosity-gap headlines), varying the pacing, and targeting specific emotional pain points. "
                    "Ensure the content has clear transition lines into the main body. Maximize save-to-reach potential by delivering high-density value breakdowns. "
                    "CRITICAL: Do NOT use any markdown formatting. Do NOT use bullet points, hyphens (-), or dashes for lists. Write in natural flowing paragraphs. "
                    "Do NOT sound like an AI. Completely avoid AI buzzwords like 'Unlock', 'Dive in', 'In today\\'s digital landscape', 'Elevate', or 'Discover'. Keep the tone conversational, authentic, and natural. "
                    "STRICT RULE: If generating multiple options, you MUST separate each option strictly with the string '---OPTION---' on a new line. Do NOT output labels like 'Option 1' or 'Variation 2'. "
                    "STRICT RULE: Output ONLY the raw post content. NEVER include conversational filler like 'Here are your posts', 'Sure', or 'You\\'re welcome!'. Start the first post immediately."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=1.1,
            presence_penalty=1.0,
            frequency_penalty=1.0
        )
        return response.choices[0].message.content
    else:
        # Fallback to standard OpenAI if needed, or raise exception
        raise ValueError("Azure OpenAI configuration is missing or invalid.")

def generate_image_for_post(prompt: str, brand_name: str = None, logo_path: str = None, website_url: str = None, logo_bytes: bytes = None, variation_index: int = 0, preferred_style: str = None) -> str:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    endpoint_url = os.environ.get("Endpoint")
    api_key = os.environ.get("Secret_Key")
    model_name = os.environ.get("Model", "gpt-image-2.5-flare")
    
    if not endpoint_url or not api_key:
        return None
        
    try:
        # Based on standard AzureOpenAI configuration for images
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-15-preview",
            azure_endpoint=endpoint_url
        )
        
        # 5 distinct visual styles & perspectives to ensure each generated card has a unique composition
        styles = [
            # 0: Minimalist Flat-lay / Knolling Desk Setup
            "Clean overhead flat-lay / knolling photography, minimalist aesthetic, natural diffuse daylight, organized premium stationery, notebook with blueprint sketches, subtle tech accessories on matte marble or warm oak surface. Top-down view.",
            # 1: Close-up Macro / Tangible Metaphor
            "Intimate macro close-up photography, shallow depth of field with gorgeous bokeh. Focus on intricate precision gears, elegant metallic hourglass, or glowing fiber-optic network connectors. Dramatic side rim lighting.",
            # 2: Modern Architectural Interior / Studio Atmosphere
            "Wide architectural commercial photography of a sleek futuristic design studio or glass boardroom, sunbeams streaming through floor-to-ceiling windows, sculptural modern furniture, ultra-clean aesthetic, airy atmosphere.",
            # 3: Sleek Tech Object / Floating Isometric Metaphor
            "Studio product photography, high-end dark luxury theme or matte ceramic texture, sharp directional spotlight, abstract geometric prism or clean glass crystal reflecting subtle data rays. Crisp high-end catalog look.",
            # 4: Editorial Graphic & Concept Still-Life
            "Contemporary magazine editorial still-life, bold creative composition, textured concrete or linen background, subtle natural shadows, artistic arrangement of physical props symbolizing growth and clarity."
        ]
        
        current_style = styles[variation_index % len(styles)]
        
        # If user has demonstrated a strong approved style preference, blend in subtle elements of it
        user_preference_guidance = ""
        if preferred_style:
            user_preference_guidance = f"User Approval Aesthetic Tendency: The user frequently approves {preferred_style}. Favor similar lighting, elegance, and premium restraint. "
        
        import uuid
        image_prompt = (
            f"Style & Composition: {current_style} "
            f"{user_preference_guidance}"
            "CRITICAL: Absolutely NO human faces or people in the image. Visuals must be strictly realistic, authentic, and tangible. "
            "NEGATIVE CONSTRAINTS: Absolutely no repetitive laptop-on-desk clichés, no cartoonish styles, no cheap 3D plastic renders, no messy multi-color gradients, no childish vectors, no distorted hands. "
            "Extract a short, powerful 3-to-4 word hook from the context and render it cleanly with elegant, minimalist, sharp modern typography integrated tastefully into the composition. "
            "DO NOT write the brand name anywhere in the image, as a logo will be overlaid later. "
            f"BATCH SEED: {uuid.uuid4()} "
            f"Topic Context: {prompt[:700]}"
        )
        
        try:
            response = client.images.generate(
                model=model_name,
                prompt=image_prompt,
                n=1,
                size="1024x1024"
            )
        except Exception as e:
            # If rejected by safety filter, fallback to a completely neutral generic prompt
            print(f"Initial image generation failed, retrying with safe fallback prompt. Error: {e}")
            try:
                fallback_prompt = (
                    f"Clean aesthetic still-life photography: {current_style}. {user_preference_guidance}"
                    "CRITICAL: Absolutely NO human faces or people. No generic cartoon gradients."
                )
                response = client.images.generate(
                    model=model_name,
                    prompt=fallback_prompt,
                    n=1,
                    size="1024x1024"
                )
            except Exception as e2:
                print(f"Fallback AI generation also failed: {e2}")
                response = None
        
        if response and response.data and len(response.data) > 0:
            img_data = response.data[0]
        else:
            # Guaranteed fallback using Pillow
            img_data = None
            
        # 1. Get raw image bytes
        image_bytes = None
        if img_data:
            if getattr(img_data, 'b64_json', None):
                image_bytes = base64.b64decode(img_data.b64_json)
            elif getattr(img_data, 'url', None):
                resp = requests.get(img_data.url)
                if resp.status_code == 200:
                    image_bytes = resp.content
        
        if not image_bytes:
            # Generate a beautiful gradient fallback image using PIL
            print("Generating fallback gradient image locally...")
            import math
            img = Image.new('RGB', (1024, 1024))
            # Create a simple blue-purple gradient
            for y in range(1024):
                for x in range(1024):
                    r = int(255 * (x / 1024.0) * 0.5)
                    g = int(255 * (y / 1024.0) * 0.3)
                    b = int(255 * (0.5 + 0.5 * (x + y) / 2048.0))
                    img.putpixel((x, y), (r, g, b))
            
            output_io = io.BytesIO()
            img.save(output_io, format="PNG")
            image_bytes = output_io.getvalue()
            
        # 2. Add Watermark and Logo
        try:
            base_img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
            
            # Composite logo if available
            logo_img = None
            if logo_bytes:
                try:
                    logo_img = Image.open(io.BytesIO(logo_bytes)).convert("RGBA")
                except Exception as b_err:
                    print(f"Failed to open logo from bytes: {b_err}")
            elif logo_path and os.path.exists(logo_path):
                try:
                    logo_img = Image.open(logo_path).convert("RGBA")
                except Exception as p_err:
                    print(f"Failed to open logo from path: {p_err}")

            if logo_img:
                try:
                    target_width = int(base_img.width * 0.15)
                    aspect_ratio = logo_img.height / logo_img.width
                    target_height = int(target_width * aspect_ratio)
                    logo_img = logo_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    
                    padding = 30
                    position = (base_img.width - target_width - padding, padding)
                    base_img.alpha_composite(logo_img, dest=position)
                except Exception as comp_err:
                    print(f"Failed to overlay logo: {comp_err}")
            
            # Add jmstech.co at the bottom
            try:
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(base_img)
                if not website_url:
                    raise Exception("No website URL provided")
                domain = website_url.replace("https://", "").replace("http://", "").strip("/")
                if not domain.startswith("www."):
                    website_text = "www." + domain
                else:
                    website_text = domain
                try:
                    font = ImageFont.truetype("arialbd.ttf", 38)
                except:
                    font = ImageFont.load_default()
                    
                bbox = draw.textbbox((0, 0), website_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                text_x = 40
                text_y = base_img.height - text_h - 40
                
                # Draw text with subtle drop shadow for readability
                shadow_offset = 2
                draw.text((text_x + shadow_offset, text_y + shadow_offset), website_text, font=font, fill=(0, 0, 0, 180))
                draw.text((text_x, text_y), website_text, font=font, fill=(255, 255, 255, 255))
            except Exception as txt_err:
                print(f"Failed to draw website text: {txt_err}")
                
            output_io = io.BytesIO()
            base_img.save(output_io, format="PNG")
            image_bytes = output_io.getvalue()
        except Exception as e:
            print(f"Image modification failed: {e}")
            
        # 3. Save the image to the storage (Azure Blob or local filesystem)
        filename = f"ai_images/generated_img_{uuid.uuid4().hex[:8]}.png"
        saved_path = default_storage.save(filename, ContentFile(image_bytes))
            
        # 4. Return the URL to access it
        return default_storage.url(saved_path)
    except Exception as e:
        print(f"Image generation failed: {e}")
        
    return None


def generate_content_with_gemini(prompt: str) -> str:
   
    from dotenv import load_dotenv
    import os
    import requests
    load_dotenv(override=True)
    api_key = os.environ.get("gemini_key")
    if not api_key:
        raise ValueError("Gemini API key (gemini_key) is missing in .env")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "system_instruction": {
            "parts": {
                # "text": "You are a top-tier expert social media manager. Write highly engaging, human-like social media posts."
                "text": (
                    "You are a top-tier Direct-Response Copywriter and Social Media Operations Director. "
                    "Write highly engaging, human-like social media posts that convert, drive engagement, and are engineered for social media feeds. "
                    "Use pattern-interrupt hooks (e.g., 3-second curiosity-gap headlines) and clear transition lines. "
                    "Maximize save-to-reach potential by delivering high-density value breakdowns."
                )
            }
        },
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    import time
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data['candidates'][0]['content']['parts'][0]['text']
        except requests.exceptions.HTTPError as e:
            if response.status_code in (429, 503) and attempt < 2:
                print(f"Retrying text generation due to {response.status_code}...")
                time.sleep(2 ** attempt)
                continue
            print(f"Error calling Gemini API for text: {e}")
            raise ValueError(f"Gemini API text generation failed: {e}")
        except Exception as e:
            print(f"Error calling Gemini API for text: {e}")
            raise ValueError(f"Gemini API text generation failed: {e}")


def generate_image_with_gemini(prompt: str, brand_name: str = None, logo_path: str = None) -> str:
    # Separate function to generate an image using Gemini's API (e.g. Imagen 3 if available on the key).
    from dotenv import load_dotenv
    import os
    import requests
    load_dotenv(override=True)
    api_key = os.environ.get("gemini_key")
    if not api_key:
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/nano-banana-pro-preview:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    import time
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            if 'candidates' in data and len(data['candidates']) > 0:
                parts = data['candidates'][0].get('content', {}).get('parts', [])
                if parts and 'inlineData' in parts[0]:
                    b64_img = parts[0]['inlineData'].get('data')
                    if b64_img:
                        import base64
                        import uuid
                        from django.conf import settings
                        
                        image_bytes = base64.b64decode(b64_img)
                        
                        # We can add logo compositing here if needed, similar to old code
                        
                        filename = f"ai_images/gemini_img_{uuid.uuid4().hex[:8]}.png"
                        saved_path = default_storage.save(filename, ContentFile(image_bytes))
                        return default_storage.url(saved_path)
            return None
        except requests.exceptions.HTTPError as e:
            if response.status_code in (429, 503) and attempt < 2:
                if hasattr(response, 'text') and 'quota' in response.text.lower():
                    break
                time.sleep(5)
                continue
            print(f"Error calling Gemini Image API: {e}")
            return "https://placehold.co/800x800/png?text=Fallback+Image+(Quota+Exceeded)"
        except Exception as e:
            print(f"Error calling Gemini Image API: {e}")
            return "https://placehold.co/800x800/png?text=Fallback+Image+(Quota+Exceeded)"
            
    return "https://placehold.co/800x800/png?text=Fallback+Image+(Quota+Exceeded)"


def generate_video_with_gemini(prompt: str) -> str:
    # Separate function to generate a video using Gemini's API.
    from dotenv import load_dotenv
    import os
    load_dotenv(override=True)
    api_key = os.environ.get("gemini_key")
    if not api_key:
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/veo-3.1-fast-generate-preview:predictLongRunning?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "instances": [
            {"prompt": prompt}
        ],
        "parameters": {
            "sampleCount": 1
        }
    }
    
    import time
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            if 'candidates' in data and len(data['candidates']) > 0:
                parts = data['candidates'][0].get('content', {}).get('parts', [])
                if parts and 'inlineData' in parts[0]:
                    b64_video = parts[0]['inlineData'].get('data')
                    if b64_video:
                        import base64
                        import uuid
                        
                        video_bytes = base64.b64decode(b64_video)
                        filename = f"ai_videos/gemini_video_{uuid.uuid4().hex[:8]}.mp4"
                        saved_path = default_storage.save(filename, ContentFile(video_bytes))
                        return default_storage.url(saved_path)
            return None
        except requests.exceptions.HTTPError as e:
            if response.status_code in (429, 503) and attempt < 2:
                if hasattr(response, 'text') and 'quota' in response.text.lower():
                    break
                time.sleep(5)
                continue
            print(f"Error calling Gemini Video API: {e} - Response: {response.text}")
            return None
        except Exception as e:
            print(f"Error calling Gemini Video API: {e}")
            return None
            
    return None
