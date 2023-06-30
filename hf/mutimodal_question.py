from transformers import pipeline
from PIL import Image
from pdf2image import convert_from_path

# Invoices
pdf = "../samples/Commercial Invoice _ 80457372.pdf"
pages = convert_from_path(pdf, 500)

for count, page in enumerate(pages):
    page.save(f'out{count}.jpg', 'JPEG')

# pipe = pipeline("document-question-answering", model="naver-clova-ix/donut-base-finetuned-docvqa")
pipe = pipeline("document-question-answering", model="impira/layoutlm-document-qa")

image = Image.open('out0.jpg')

res = {
    "purchase_amount": pipe(image=image, question="What is the purchase amount?"),
    "item": pipe(image=image, question="What is the item title?"),
    "item_dimensions": pipe(image=image, question="What are the items dimensions?"),
    "item_width": pipe(image=image, question="What is the width of the item?"),
    "item_height": pipe(image=image, question="What is the height of the item?"),
    "item_weight": pipe(image=image, question="What is the weight of the item?"),
    "company_name": pipe(image=image, question="What is the company name?"),
    "company_eori": pipe(image=image, question="What is the company eori?"),
    "company_address": pipe(image=image, question="What is the company address?")
}

print('res =>', res)
## [{'answer': '20,000$'}]
