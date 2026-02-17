from fastapi import FastAPI

app=FastAPI()



indian_places = {
    'delhi': ['Red Fort', 'Qutub Minar', 'India Gate'],
    'mumbai': ['Gateway of India', 'Marine Drive', 'Elephanta Caves'],
    'jaipur': ['Hawa Mahal', 'Amber Fort', 'City Palace'],
    'varanasi': ['Kashi Vishwanath Temple', 'Ghosts of Ganges', 'Sarnath'],
    'goa': ['Baga Beach', 'Calangute Beach', 'Dudhsagar Falls']
}

@app.get("/get_items/{name}")
async def hello(name):
    return indian_places.get(name)
