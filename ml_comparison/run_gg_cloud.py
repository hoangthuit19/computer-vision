from google.cloud import vision

def detect_text(path):
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        print("Detected text:")
        print(texts[0].description)  # full text
    else:
        print("No text detected.")

    if response.error.message:
        raise Exception(f"{response.error.message}")

# Test
detect_text("")
