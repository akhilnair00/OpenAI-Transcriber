###########################################################

# Test 1: Utilizing GPT2

# from transformers import pipeline

# text_input = "Who are you?"

# pipe = pipeline("text-generation", model="gpt2")

# output = pipe(text_input)
# print(output)

# Results: Run's poorly.

###########################################################

# Test 2: Utilizing whisper-1 and synthia

# import sys
# from langchain_community.llms import LlamaCpp
# import openai
# import os

# verbose = False

# llm = LlamaCpp(
#     model_path="C:/Users/akhil/OneDrive/Documents/Creative Endeavour/OpenAI Transcriber/synthia-7b-v2.0-16k.Q4_K_M.gguf",
#     n_ctx=4096,
#     n_gpu_layers=32,
#     n_batch=1024,
#     f16_kv=True,
#     verbose=verbose,
# )

# openai.api_key = os.getenv('OPENAI_API_KEY')

# # utilize whisper-1 to acquire the text from audio file
# with open("C:/Users/akhil/OneDrive/Documents/Creative Endeavour/OpenAI Transcriber/jishandmetalkingtest.mp3", "rb") as audio_file:
#     transcription = openai.Audio.transcribe("whisper-1", audio_file)

# # print text for user to see
# print(transcription['text'])

# question = 'Please summarize the following text as meeting notes: ' + transcription['text']

# # print(question)

# output = llm(
#     question,
#     max_tokens=4096,
#     temperature=0.2,
#     top_p=0.1
# )

# print({output})

# Results: Will have to do some sort of fine tuning to fully utilize this.

###########################################################

# Test 3: Utilizing Synthia with just the text prompt

# import sys
# from langchain_community.llms import LlamaCpp

# verbose = False

# llm = LlamaCpp(
#     model_path="C:/Users/akhil/OneDrive/Documents/Creative Endeavour/OpenAI Transcriber/synthia-7b-v2.0-16k.Q4_K_M.gguf",
#     n_ctx=4096,
#     n_gpu_layers=32,
#     n_batch=1024,
#     f16_kv=True,
#     verbose=verbose,
# )

# transcript = "Um, I had a pretty chill day, um, obviously I was feeling a bit under the weather today, but it was nice that I got to sit around, watch some Netflix, and just kind of recover. That's good, that's good. Well, I'm glad you at least got to relax a little bit on your long weekend. Yeah, um, what were you up to today? I was bouldering with Rudrik, that was fun, and it was a lot of fun. I went bouldering with Rudrik, that was fun, and it was actually a great workout, like, no joke. It was super quick, because we went at like 7.30, but regardless, it was a good workout. And then after that, we just talked, and it was a really nice talk, it was like, very chill. He just drove me around, and, um, yeah, it was a good time. Nice, what things you guys talked about? Nothing, just like, his life and my life, and how things are going and stuff, you know? Cool, cool. But yeah, we're going to boulder sometime this week as well. I was about to invite him to, like, quadruple with me, you, Zohar, and Tyler. I was going to invite him to, like, fifth wheel. Because he had asked me, he was like, are you planning on climbing tomorrow? And I didn't want to say, like, lie. But then he never brought it up again, so I didn't get to. Oh, I mean, I wouldn't mind, and I don't think Zohar and Tyler would mind. Yeah, I don't think so either. I mean, regardless, he can just go by himself, right? Right, right. But overall, it was a great climbing sesh, great talking sesh. I think I had a fun time. That's great. Do you feel like you climbed some good projects today? Mm-hmm. They've actually, like, genuinely, they've actually replaced some of the walls now. So there's some good V2s and V3s in there that are, like, testy. Oh, nice. Okay, well, hopefully tomorrow, um, us four can try some new ones. What time are we going tomorrow? So I told Zohar 1 p.m. for us to go. Okay, 1 p.m. actually sounds pretty good. I guess I'll eat lunch afterwards, because I really don't want to eat lunch beforehand. Yeah, I'm sorry it's an awkward timing. My mom and I are going to get brunch somewhere. Yeah, you said the gym closes at what time? Four o'clock. Oh, darn. Okay, that makes sense then. I was going to suggest we eat later. Go later? Yeah, if we do need to change it, we can. Zohar said they are open, they're free the whole day. Okay, cool. All right. Thanks, babe. Love you. Yeah, love you, too."

# question = 'Please summarize the following text as meeting notes: \n' + transcript

# print(question)

# output = llm.invoke(
#     question,
#     max_tokens=4096,
#     temperature=0.2,
#     top_p=0.1
# )

# print(output)

# Results: Synthia is not meant for long form text and has a weird personality.

###########################################################

# Test 4: utilizing pipeline API

import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-9b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda", 
)

prompt = "Please summarize the following text as meeting notes: \n"

transcript = "Um, I had a pretty chill day, um, obviously I was feeling a bit under the weather today, but it was nice that I got to sit around, watch some Netflix, and just kind of recover. That's good, that's good. Well, I'm glad you at least got to relax a little bit on your long weekend. Yeah, um, what were you up to today? I was bouldering with Rudrik, that was fun, and it was a lot of fun. I went bouldering with Rudrik, that was fun, and it was actually a great workout, like, no joke. It was super quick, because we went at like 7.30, but regardless, it was a good workout. And then after that, we just talked, and it was a really nice talk, it was like, very chill. He just drove me around, and, um, yeah, it was a good time. Nice, what things you guys talked about? Nothing, just like, his life and my life, and how things are going and stuff, you know? Cool, cool. But yeah, we're going to boulder sometime this week as well. I was about to invite him to, like, quadruple with me, you, Zohar, and Tyler. I was going to invite him to, like, fifth wheel. Because he had asked me, he was like, are you planning on climbing tomorrow? And I didn't want to say, like, lie. But then he never brought it up again, so I didn't get to. Oh, I mean, I wouldn't mind, and I don't think Zohar and Tyler would mind. Yeah, I don't think so either. I mean, regardless, he can just go by himself, right? Right, right. But overall, it was a great climbing sesh, great talking sesh. I think I had a fun time. That's great. Do you feel like you climbed some good projects today? Mm-hmm. They've actually, like, genuinely, they've actually replaced some of the walls now. So there's some good V2s and V3s in there that are, like, testy. Oh, nice. Okay, well, hopefully tomorrow, um, us four can try some new ones. What time are we going tomorrow? So I told Zohar 1 p.m. for us to go. Okay, 1 p.m. actually sounds pretty good. I guess I'll eat lunch afterwards, because I really don't want to eat lunch beforehand. Yeah, I'm sorry it's an awkward timing. My mom and I are going to get brunch somewhere. Yeah, you said the gym closes at what time? Four o'clock. Oh, darn. Okay, that makes sense then. I was going to suggest we eat later. Go later? Yeah, if we do need to change it, we can. Zohar said they are open, they're free the whole day. Okay, cool. All right. Thanks, babe. Love you. Yeah, love you, too."

messages = [
    {"role": "user", "content": prompt + transcript},
]

outputs = pipe(messages, max_new_tokens=256)
assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
print(assistant_response)