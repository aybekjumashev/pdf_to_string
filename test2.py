# Diagnostika strategiyasini to'g'ridan-to'g'ri ishlatish
import fitz
import pytesseract
from PIL import Image
import io

# PDF ni ochish
doc = fitz.open("file2.pdf")

content = ''

for page in doc:
    print(f"Page {page.number} size: {page.rect}")

    # Xuddi diagnostikadagi kabi
    mat = fitz.Matrix(5.5, 5.5)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_data))

    # OCR - diagnostikada ishlagan
    text = pytesseract.image_to_string(
        image,
        lang='uzb+rus+eng',  # yoki 'rus+eng'
        config='--psm 3'
    )

    content += text + '\n'


with open("file2.txt", "w", encoding="utf-8") as f:
    f.write(content)

# PDF faylini yopish
doc.close()
