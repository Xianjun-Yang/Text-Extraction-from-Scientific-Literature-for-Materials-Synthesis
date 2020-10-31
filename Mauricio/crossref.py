import requests
import json

def jprint(obj):
    # create a formatted string of the Python JSON object
    text = json.dumps(obj, sort_keys=True, indent=4)
    print(text)

url = 'http://api.crossref.org/works/10.1038/s41586-019-1335-8'
writefile = open('./newfile.pdf', 'wb')

headers = {
        'Accept': 'application/json'
      }

response = json.loads(requests.get(url, headers=headers).text)
url = response['message']['link'][0]['URL']
app_type = str(response['message']['link'][0]['content-type'])

if app_type in ['application/pdf', 'unspecified']:
    headers['Accept'] = 'application/pdf'
    r = requests.get(url, stream=True, headers=headers)
    if r.status_code == 200:
        for chunk in r.iter_content(2048):
            writefile.write(chunk)