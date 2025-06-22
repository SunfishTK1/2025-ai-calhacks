# %%
import pandas as pd
import json
import os

from dotenv import load_dotenv
load_dotenv()

from openai import AzureOpenAI
from pydantic import BaseModel, HttpUrl


# %%
df = pd.read_csv("raw_data/Business_Licenses_20250621.csv") 
df.head()

# %%
# build the schema for businesses
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Dict, Optional


class Hours(BaseModel):
    mon_sun: str = Field(..., min_length=1)


class Vibe(BaseModel):
    crowd: str
    atmosphere: str
    events: List[str]


class Reviews(BaseModel):
    yelp_rating: float
    restaurantguru_rating: float
    birdeye_rating: float
    common_feedback: List[str]
    sample_quotes: List[str]


class SocialMedia(BaseModel):
    instagram: str
    latest_event_post: str


class Business(BaseModel):
    name: str
    address: str
    phone: str
    website: HttpUrl
    hours: Hours
    established: Optional[str]
    type: str
    menu_highlights: List[str]
    vibe: Vibe
    reviews: Reviews
    parking: str
    payment: List[str]
    wifi: str
    delivery: str
    social_media: SocialMedia


business_info_schema = {
    "type": "function",
    "function": {
        "name": "get_business_info",
        "description": "Structured business metadata for businesses in Berkeley, CA.",
        "parameters": Business.model_json_schema()
    }
}

# %%
shortened_df = df[["BusDesc", "B1_BUSINESS_NAME", "B1_FULL_ADDRESS"]]

endpoint = os.getenv("ENDPOINT_URL", "https://2025-ai-hackberkeley.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "o4-mini")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-12-01-preview",  # Use a valid API version
)

def get_business_info(business_name):
    response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a search agent, and your task is to collect more information based on a given information of a business. use web search. access reviews about the business if you can. access big chunk of texts regarding what they serve or who they serve for. include things such as vibes about the place that cannot be easily found through traditional searh engines. The user will provide the business information, and you will limit your answer to a json data format with clear data output regarding any information you can find.",
                    },
                    {
                        "role": "user",
                        "content": f"{business_name}",
                    },
                ],
            )
    
    raw_output = response.choices[0].message.content
    cleaned = raw_output.strip().removeprefix("```json").removesuffix("```").strip()

    return cleaned

# %%
for index, row in shortened_df[:100].iterrows():
    business_query = row['B1_BUSINESS_NAME'] + " " + row['B1_FULL_ADDRESS']
    print(f"Processing {business_query}...")
    try:
        response = get_business_info(business_query)
        print(f"Response for {business_query}: {response}")
    except Exception as e:
        print(f"Error processing {business_query}: {e}")

    shortened_df.at[index, 'Business Detailed Data'] = response

# %%
shortened_df.to_csv("augmented_data/business_info_first_100.csv", index=False)

# %%



