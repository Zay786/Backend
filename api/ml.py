from io import BytesIO
import base64
import random
import uuid

from fastapi import FastAPI
from pydantic import BaseModel
from reportlab.pdfgen import canvas

app = FastAPI()


class QuoteRequest(BaseModel):
    name: str
    company: str
    email: str
    origin: str
    destination: str
    commodity: str
    weight: float
    service: str


@app.post("/")
@app.post("/generate")
@app.post("/api/ml/generate")
def generate_quote(data: QuoteRequest):
    base_price = random.randint(1000, 5000)
    price = base_price + (data.weight * 50)

    file_name = f"quotation_{uuid.uuid4().hex}.pdf"
    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer)

    pdf.drawString(100, 750, "TOM & JERRY LOGISTICS QUOTATION")
    pdf.drawString(100, 720, f"Customer: {data.name}")
    pdf.drawString(100, 700, f"Company: {data.company}")
    pdf.drawString(100, 680, f"Origin: {data.origin}")
    pdf.drawString(100, 660, f"Destination: {data.destination}")
    pdf.drawString(100, 640, f"Commodity: {data.commodity}")
    pdf.drawString(100, 620, f"Weight: {data.weight} Tons")
    pdf.drawString(100, 600, f"Service: {data.service}")
    pdf.drawString(100, 560, f"Estimated Price: ${price}")

    pdf.save()
    pdf_buffer.seek(0)

    return {
        "price": price,
        "pdf_base64": base64.b64encode(pdf_buffer.getvalue()).decode("utf-8"),
        "pdf_file_name": file_name,
    }
