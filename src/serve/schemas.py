from pydantic import BaseModel

class FlightInput(BaseModel):
    MONTH: int
    DAY: int
    DAY_OF_WEEK: int
    AIRLINE: str
    ORIGIN_AIRPORT: str
    DESTINATION_AIRPORT: str
    SCHEDULED_DEPARTURE: int
    DISTANCE: float
    SCHEDULED_TIME: float
