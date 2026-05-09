from io import BytesIO
import base64
import math
import uuid

from fastapi import FastAPI
from pydantic import BaseModel, Field
from reportlab.pdfgen import canvas

app = FastAPI()
MIN_HISTORY_FOR_KNN = 15


class HistoricalQuote(BaseModel):
    origin: str
    destination: str
    commodity: str
    weight_tons: float
    service_type: str
    predicted_price: float


class QuoteRequest(BaseModel):
    name: str
    company: str | None = ""
    email: str
    origin: str
    destination: str
    commodity: str
    weight: float
    service: str
    historical_quotes: list[HistoricalQuote] = Field(default_factory=list)


def _build_categories(history: list[HistoricalQuote], request: QuoteRequest) -> dict[str, list[str]]:
    return {
        "origins": sorted({quote.origin for quote in history} | {request.origin}),
        "destinations": sorted({quote.destination for quote in history} | {request.destination}),
        "commodities": sorted({quote.commodity for quote in history} | {request.commodity}),
        "services": sorted({quote.service_type for quote in history} | {request.service}),
    }


def _encode_features(
    *,
    origin: str,
    destination: str,
    commodity: str,
    weight: float,
    service: str,
    categories: dict[str, list[str]],
    max_weight: float,
) -> list[float]:
    safe_max_weight = max(max_weight, 1.0)
    features = [weight / safe_max_weight]

    for value in categories["origins"]:
        features.append(1.0 if origin == value else 0.0)

    for value in categories["destinations"]:
        features.append(1.0 if destination == value else 0.0)

    for value in categories["commodities"]:
        features.append(1.0 if commodity == value else 0.0)

    for value in categories["services"]:
        features.append(1.0 if service == value else 0.0)

    return features


def _euclidean_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _fallback_formula(request: QuoteRequest) -> tuple[float, dict]:
    service_multiplier = {
        "Air Freight": 1.8,
        "Sea Freight": 1.15,
        "Land Transport": 1.0,
    }
    commodity_multiplier = {
        "Sulphur": 1.12,
        "Copper": 1.3,
        "Maize": 0.95,
        "Wheat": 0.9,
    }
    port_coordinates = {
        "Tanzania": (0.0, 0.0),
        "Walvisbay": (3.2, 1.4),
        "Shanghai": (8.4, 7.1),
        "Busan": (8.8, 7.7),
        "Port Louis": (2.1, 2.0),
        "Mumbai": (5.9, 4.5),
        "London": (9.1, 9.0),
        "Paris": (8.8, 8.7),
    }

    origin = port_coordinates.get(request.origin, (0.0, 0.0))
    destination = port_coordinates.get(request.destination, (4.0, 4.0))
    route_distance = math.dist(origin, destination)

    base_charge = 850.0
    weight_charge = max(float(request.weight), 1.0) * 115.0
    route_charge = route_distance * 180.0
    service_charge = service_multiplier.get(request.service, 1.0)
    commodity_charge = commodity_multiplier.get(request.commodity, 1.0)

    price = (base_charge + weight_charge + route_charge) * service_charge * commodity_charge

    return price, {
        "algorithm": "fallback_formula",
        "records_used": 0,
        "factors": {
            "base_charge": base_charge,
            "weight_charge": round(weight_charge, 2),
            "route_distance": round(route_distance, 2),
            "route_charge": round(route_charge, 2),
            "service_multiplier": service_charge,
            "commodity_multiplier": commodity_charge,
        },
    }


def predict_price(request: QuoteRequest) -> tuple[float, dict]:
    history = [
        quote
        for quote in request.historical_quotes
        if quote.predicted_price is not None and quote.weight_tons is not None
    ]

    if len(history) < MIN_HISTORY_FOR_KNN:
        price, model_details = _fallback_formula(request)
        model_details["records_available"] = len(history)
        model_details["min_history_for_knn"] = MIN_HISTORY_FOR_KNN
        return price, model_details

    categories = _build_categories(history, request)
    max_weight = max([request.weight] + [float(quote.weight_tons) for quote in history])

    request_vector = _encode_features(
        origin=request.origin,
        destination=request.destination,
        commodity=request.commodity,
        weight=float(request.weight),
        service=request.service,
        categories=categories,
        max_weight=max_weight,
    )

    scored_quotes = []
    for quote in history:
        quote_vector = _encode_features(
            origin=quote.origin,
            destination=quote.destination,
            commodity=quote.commodity,
            weight=float(quote.weight_tons),
            service=quote.service_type,
            categories=categories,
            max_weight=max_weight,
        )
        distance = _euclidean_distance(request_vector, quote_vector)

        # Give more influence to highly similar historical quotations while
        # still allowing nearby records to contribute.
        similarity_weight = 1.0 / (distance + 0.05)
        scored_quotes.append(
            {
                "distance": distance,
                "weight": similarity_weight,
                "price": float(quote.predicted_price),
            }
        )

    scored_quotes.sort(key=lambda item: item["distance"])
    neighbors = scored_quotes[: min(5, len(scored_quotes))]

    weighted_total = sum(item["price"] * item["weight"] for item in neighbors)
    total_weight = sum(item["weight"] for item in neighbors)
    predicted_price = weighted_total / total_weight if total_weight else neighbors[0]["price"]

    return predicted_price, {
        "algorithm": "knn_regression",
        "records_available": len(history),
        "records_used": len(neighbors),
        "min_history_for_knn": MIN_HISTORY_FOR_KNN,
        "nearest_distances": [round(item["distance"], 4) for item in neighbors],
    }


@app.post("/")
@app.post("/generate")
@app.post("/api/ml/generate")
def generate_quote(data: QuoteRequest):
    predicted_price, model_details = predict_price(data)
    price = round(predicted_price, 2)
    model_algorithm = model_details.get("algorithm", "unknown")

    file_name = f"quotation_{uuid.uuid4().hex}.pdf"
    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer)

    # Title in navy blue
    pdf.setFillColorRGB(0, 0, 0.5)  # Navy blue
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(100, 750, "TOM & JERRY Logistics")

    # Subtitle
    pdf.setFillColorRGB(0, 0, 0)  # Black
    pdf.setFont("Helvetica", 14)
    pdf.drawString(100, 720, "Quotation")

    # Customer details
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(100, 680, "Customer Information:")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(120, 660, f"Name: {data.name}")
    pdf.drawString(120, 640, f"Company: {data.company or 'N/A'}")

    # Shipment details
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(100, 600, "Shipment Details:")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(120, 580, f"Origin: {data.origin}")
    pdf.drawString(120, 560, f"Destination: {data.destination}")
    pdf.drawString(120, 540, f"Commodity: {data.commodity}")
    pdf.drawString(120, 520, f"Weight: {data.weight} Tons")
    pdf.drawString(120, 500, f"Service: {data.service}")

    # Price
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(100, 460, f"Estimated Price: ${price}")

    # Note
    pdf.setFont("Helvetica", 10)
    note_text = (
        "Please note that this is just a quotation and not the fixed price. "
        "The actual price may have a little variation. Kindly get into contact "
        "with our team on customerservice@tomjerry.com to further discuss about "
        "turning this quotation opportunity into a reality!"
    )
    pdf.drawString(100, 420, note_text[:60])
    pdf.drawString(100, 405, note_text[60:120])
    pdf.drawString(100, 390, note_text[120:180])
    pdf.drawString(100, 375, note_text[180:240])
    pdf.drawString(100, 360, note_text[240:])

    # Model used in light grey
    pdf.setFillColorRGB(0.7, 0.7, 0.7)  # Light grey
    pdf.setFont("Helvetica", 8)
    pdf.drawString(100, 320, f"Model Used: {model_algorithm}")

    pdf.save()
    pdf_buffer.seek(0)

    return {
        "price": price,
        "pdf_base64": base64.b64encode(pdf_buffer.getvalue()).decode("utf-8"),
        "pdf_file_name": file_name,
        "model_details": model_details,
    }
