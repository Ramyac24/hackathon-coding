import google.generativeai as genai

# Configure the API
genai.configure(api_key="AIzaSyCOO5dF5Bh6yQtdcF-ob30M0YQjku8CDBo")

# Choose your model
model = genai.GenerativeModel("gemini-2.5-pro")

# Send a video
response = model.generate_content([
    "Analyze this video and describe what's happening.",
    {
        #"mime_type": "image/png",
        #"data": open("Screenshot_sample.png", "rb").read(),
        "mime_type": "video/mp4",
        "data": open("sample.mp4", "rb").read()
    }
])

print(response.text)



