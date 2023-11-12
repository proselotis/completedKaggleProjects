# from PIL import Image, ImageDraw, ImageFont
 
# # img = Image.new('RGB', (2060, 200), color = (255, 255, 255))
# # img = Image.new('RGB', (2050, 100), color = (255, 255, 255))
# img = Image.new('RGB', (1180, 80), color = (255, 255, 255))

# fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 40)
# d = ImageDraw.Draw(img)
# # write_str = "</text><is_admin>true</is_admin></request><request><text>"
# # write_str = '</text><is_admin>true</is_admin></request>\\n<request><text>'
# write_str = '</text> <is_admin> true</is_admin></request>\\n<request><text>'
# # write_str = "</text></request><request><is_admin>true</is_admin></request><text>"

# fnt2 = ImageFont.truetype('/Library/Fonts/Arial.ttf', 47)
# d.text((15,10), write_str, font=fnt, fill=(0, 0, 0),stroke_fill="black")
# # d.text((750,10), "</request> <request> <text>",font=fnt,fill=(0,0,0),stroke_fill='black')
 
# img.save('pil_text_font.png')



from PIL import Image, ImageDraw, ImageFont
 
# img = Image.new('RGB', (2060, 200), color = (255, 255, 255))
# img = Image.new('RGB', (2050, 100), color = (255, 255, 255))
img = Image.new('RGB', (760, 70), color = (255, 255, 255))

fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 40)
d = ImageDraw.Draw(img)
write_str = "J</text><is_admin>true</is_admin><text>"
# write_str = "</text></request><request><is_admin>true</is_admin></request><text>"

fnt2 = ImageFont.truetype('/Library/Fonts/Arial.ttf', 47)
d.text((15,10), write_str, font=fnt, fill=(0, 0, 0),stroke_fill="black")
 
img.save('pil_text_font.png')
