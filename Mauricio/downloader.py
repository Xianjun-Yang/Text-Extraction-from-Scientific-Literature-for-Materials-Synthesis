import requests
import json

'''This file will contain script to automate paper downloads.
Below is an example of how it currently works. 
MUST USE UCSB VPN
'''
base_url = 'http://api.crossref.org/works/'
doi = '10.1038/s41586-019-1335-8'
url = base_url + doi
writefile = open('./paper_example.pdf', 'wb')

headers = {
        'Accept': 'application/json'
      }

response = json.loads(requests.get(url, headers=headers).text)
url = response['message']['link'][0]['URL']
app_type = str(response['message']['link'][0]['content-type'])

if app_type in ['application/pdf', 'unspecified']:
    headers['Accept'] = 'application/pdf'
    r = requests.get(url, stream=True, headers=headers)
    print(r.status_code)
    if r.status_code == 200:
        for chunk in r.iter_content(2048):
            writefile.write(chunk)