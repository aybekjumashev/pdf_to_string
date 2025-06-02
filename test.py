from pdf2string import PDFToString

# Optimallashtirilgan sozlamalar bilan test
converter = PDFToString(
    ocr_language='uzb+rus+eng'
)

result = converter.convert_pdf("file2.pdf")

print(f"Quality Score: {result['quality_score']:.1f}/100")
print(f"Text length: {len(result['text'])} characters")
print(f"Pages: {len(result['pages'])}")

# Har bir sahifani alohida ko'rish
for i, page_text in enumerate(result['pages']):
    print(f"\n=== PAGE {i+1} ===")
    print(page_text[:800])  # Birinchi 800 belgi
    print("...")

# Natijani faylga saqlash
converter.save_result(result, "final_result.txt")
print("Result saved to final_result.txt")