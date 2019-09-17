import requests

EMAIL = "ierezell@gmail.com"
PWD = "@Locryen96"


def list_domains():
    req = requests.get(
        f"https://api.freenom.com/v2/domain/list?email={EMAIL}&password={PWD}")
    print(req.text)


list_domains()
