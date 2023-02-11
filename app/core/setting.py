import os
from pydantic import BaseSettings
from starlette.config import Config


config  = Config("app/environment/environment.env")

class Setting(BaseSettings):
    environment: str = os.getenv("BUILD_ENVIRONMENT", "DEV")
    print("========= environment =========: ", environment)
    if environment != "":
        config = Config("app/environment/environment.{}env".format(environment))
    WIDTH = config("width", default= 48)




def get_setting():
    return Setting()