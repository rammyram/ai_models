from PIL import ImageFont, ImageDraw, Image

im_pil = Image.new("RGB",[320,320])
m_text = u"ひらがな - Hiragana, 히라가나"
draw = ImageDraw.Draw(im_pil)
u_font = ImageFont.truetype("arial-unicode-ms.ttf", 15)
draw.text((50, 50), m_text, font=u_font, fill=(0, 0, 255))
im_pil.save("a.png")
