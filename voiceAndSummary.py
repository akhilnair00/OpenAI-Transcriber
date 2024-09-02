import openai
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

# utilize whisper-1 to acquire the text from audio file
with open("C:/Users/akhil/OneDrive/Documents/Creative Endeavour/OpenAI Transcriber/jishandmetalkingtest.mp3", "rb") as audio_file:
    transcription = openai.Audio.transcribe("whisper-1", audio_file)

# print text for user to see
print(transcription['text'])

# utilize gpt 4o mini so i can see the response 
response = openai.ChatCompletion.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": "You will summarize the conversation to be utilized as meeting notes."},
    {"role": "user", "content": transcription['text']}
  ]
)

message_content = response['choices'][0]['message']['content']
print("Summary:", message_content)
