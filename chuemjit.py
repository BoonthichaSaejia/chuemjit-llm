import os
from pinecone import Pinecone
import numpy as np
from openai import OpenAI
from FlagEmbedding import BGEM3FlagModel
import config

model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=False)

api_key = config.PINECONE_API_KEY
pc = Pinecone(api_key=api_key)
index = pc.Index("isan")



def embed_text(texts):
  output = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
  return output['dense_vecs'] , output

def hybrid_scale(dense, indices,values, alpha: float):
  """Hybrid vector scaling using a convex combination

  alpha * dense + (1 - alpha) * sparse

  Args:
      dense: Array of floats representing
      sparse: a dict of `indices` and `values`
      alpha: float between 0 and 1 where 0 == sparse only
              and 1 == dense only
  """
  if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    # Scale sparse and dense vectors to create hybrid search vectors
  hsparse = {
      'indices': indices,
      'values':[float(v * (1 - alpha)) for v in values],
  }
  hdense = [v * alpha for v in dense]
  return hdense, hsparse

def get_ans(query,alpha):
  dense ,sparse = embed_text(query)
  dense = dense.tolist()
  indice = [int(key) for key in sparse['lexical_weights'].keys()]
  values = list(sparse['lexical_weights'].values())

  hdense, hsparse = hybrid_scale(dense, indice,values, alpha=alpha)

  num_topK = 10

  result = index.query(
      top_k=num_topK,
      vector=hdense,
      sparse_vector=hsparse,
      include_metadata=True,
      namespace='isan-srtarter-pack'
  )
  prompt = ""
  for i in range(len(result['matches'])):
    thai_central = result['matches'][i]['metadata']['textthai']
    thai_isan = result['matches'][i]['metadata']['textisan']
    prompt = prompt+"\n"+ str(i +1 ) + ".) ไทยกลาง: " + thai_central + " ,อีสาน: " + thai_isan
  return prompt


client = OpenAI(
    api_key=config.TYPHOON_API_KEY,
    base_url="https://api.opentyphoon.ai/v1",
)

def gen_text(promt):
  stream = client.chat.completions.create(
      model="typhoon-instruct",
      messages=[
          {
              "role": "user",
              "content": prompt,
          }
      ],
      max_tokens=300,
      temperature=0.8,
      top_p=1,
      stream=False,
  )

  return stream.choices[0].message.content

# Function to replace text based on dictionary
def replace_text(text):
  translation_dict = {
    "ไอติม": "กะแลม",
    "ไอศกรีม": "กะแลม",
    "แอบ": "จอบ",
    "อาการท้องร่วงอย่างรุนแรง": "ขี้ซุ๊",
    "อะไรหรอ": "หยังน้อ",
    "เหมือน": "คือจั้ง",
    "สำออย": "ขี้โยย",
    "สองวันก่อน": "มื้อซืน",
    "สวม": "กวม",
    "ส้มตำ": "ตำบักหุ่ง",
    "สนุก": "ม่วน",
    "สนิม": "ขี้เมี่ยง",
    "สดใส": "อ่องต่อง",
    "วัน": "มื้อ",
    "โรคหอบหืด": "ขี้กะยือ",
    "เรือน": "เฮือน",
    "บ้าน": "เฮือน",
    "ที่อยู่อาศัย": "เฮือน",
    "รู้": "ฮู้",
    "รีบ": "ฟ้าว",
    "รัก": "ฮัก",
    "ร้อน": "ฮ้อน",
    "รองเท้า": "เกิบ",
    "ยางของต้นพลวง": "ขี้โค้",
    "ยังไง": "จั่งได๋",
    "ไม่": "บ่",
    "ไหม": "บ่",
    "แม่ยาย": "แม่เถ้า",
    "แม่น้ำโขง": "แม่ของ",
    "แมงป่อง": "แมงงอด",
    "เมื่อวาน": "มื้อวาน",
    "เมื่่อ": "เทือ",
    "มิน่า": "กะหยอนว่า",
    "มิน่าล่ะ": "กะหยอนว่า",
    "มิดด้าม": "จำปอก",
    "มาก": "หลาย",
    "มะรืน": "มื้อฮือ",
    "มรดก": "มูลมัง",
    "เพราะ": "ย้อน",
    "พูดมาก": "เว่าหลาย",
    "พูด": "เว้า",
    "พี่ชาย": "อ้าย",
    "พี่": "อ้าย",
    "พ่อตา": "พ่อเถ้า",
    "พรุ่งนี้": "มื้ออื่น",
    "พยาธิ": "ขี้กะตืก",
    "ผู้หญิง": "แม่ญิง",
    "มอง": "เบิ่ง" ,
    "บ้ายอ": "ขี้มักย้อง",
    "น้ำพริก": "ป่น",
    "น้ำครำ": "ขี้ซีก",
    "นับประสาอะไร": "กะเสินว่ะ",
    "นะคะ": "เด้อค่ะ",
    "นะ": "เด้อ",
    "นอนบน": "นอนเกียจ",
    "เธอ": "เจ้า",
    "คุณ": "เจ้า",
    "ทิ้ง": "ถิ่ม",
    "ทำ": "เฮ็ด",
    "ทั่วโลก": "กงโลก",
    "ถึงฟ้า": "คุงฟ้า",
    "ใต้ถุน": "ใต้หล่าง",
    "แต่งตัว": "แต่งโต",
    "ตะขาบ": "ขี้เข็บ",
    "ตรวจ": "กวด",
    "ตรวจตรา": "กวด",
    "ได้ดีดั่งใจ": "คักใจ",
    "สมใจ": "คักใจ",
    "ถูกใจ": "คักใจ",
    "ดีมากๆ": "คักขนาด",
    "เยี่ยมมากๆ": "คักขนาด",
    "ดีสุดๆ": "คักขนาด",
    "สุดยอด": "คักขนาด",
    "ดิน": "ขี้ดิน",
    "ก้อนดิน": "ขี้ดิน",
    "ซุกซน": "ขี้ดื้อขี้มึน",
    "ใช่จริงหรือ": "แม่นอยู่บ้อ",
    "ใช่": "แม่น",
    "ชอบมาก": "มักหลาย",
    "เฉิ่ม": "กะเลิงเบิ๊บ",
    "ม้าดีดกระโหลก": "กะเลิงเบิ๊บ",
    "กะโหลกกะลา": "กะเลิงเบิ๊บ",
    "ฉัน": "ข่อย",
    "ผม": "ข่อย",
    "ดิฉัน": "ข่อย",
    "กระผม": "ข่อย",
    "จิ้งเหลน": "ขี้โก๋",
    "จะไป": "สิไป",
    "จริงๆ": "อีหลี",
    "แน่": "อีหลี",
    "ใคร": "ไผ",
    "คิดเสียดาย": "สานอ",
    "คิดไม่ถึง": "คิดบ่ซอด",
    "คิดไม่ออก": "คิดบ่ซอด",
    "คิดถึง": "คิดฮอด",
    "นึกถึง": "คิดฮอด",
    "นึกขึ้นได้": "คิดฮอด",
    "ฉุกคิดขึ้นได้": "คิดฮอด",
    "คาดคะเน": "กะเอา",
    "คางคก": "ขี้กะตู่",
    "คนอื่น": "เผิ่น",
    "เขา": "เผิ่น",
    "คงจะ": "คือสิ",
    "ขี้เหร่": "ขี้ฮ้าย",
    "หน้าเกลียด": "ขี้ฮ้าย",
    "ไม่สวย": "ผู้ฮ้าย",
    "ไม่หล่อ": "ผู้ฮ้าย",
    "หน้าตาไม่ดี": "ผู้ฮ้าย",
    "ขี้ฟัน": "ขี่แข่ว",
    "ขี้เกียจ": "ขี้ค้าน",
    "ขยะแขยง": "ขี้เดียด",
    "รังเกียจ": "ขี้เดียด",
    "ไม่ชอบ": "ขี้เดียด",
    "ขนมจีน": "ข้าวปุ้น",
    "ใกล้": "ม่อ",
    "เกษมสุข": "เกษิม",
    "เกลียด": "ซัง",
    "กิ้งก่า": "กะปอม",
    "กำไลมือ": "กล้องแขน",
    "กำไลแขน": "กล้องแขน",
    "กำไลขา": "กล้องขา",
    "กองฟาง": "กองเฟือง",
    "กองเต็มพื้น": "กองอ้อกยอก",
    "กลางวัน": "กลางเว็น",
    "กลัว": "ย่าน",
    "กล้วยน้ำว้า": "ก้วยออง",
    "กระบอกตา": "กระโบ่งตา",
    "ก็ช่าง": "กะหย่า",
    "ช่างเถอะ": "กะหย่า",
    "ช่างมัน": "กะหย่า",
    "ก็": "กะ",
    "เรา":"เฮา",
    "เห็น":"เบิ่ง",
    "ทุกคน":"หมู่เฮา"
}
  for thai_word, isan_word in translation_dict.items():
      if isinstance(thai_word, list):
          for word in thai_word:
              text = text.replace(word, isan_word)
      else:
          text = text.replace(thai_word, isan_word)
  return text