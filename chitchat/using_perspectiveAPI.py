"""
======================================================================
USING_PERSPECTIVEAPI ---

RUNNING perspective API for safety evaluation.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2022, ZiLiang, all rights reserved.
    Created: 21 七月 2022
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random

from googleapiclient import discovery
import json

class PerspectiveAPI:
    def __init__(self):
        # API_KEY = 'copy-your-api-key-here'
        # API_KEY = 'AIzaSyAT-Qk0HNqI63fzUEgSCfD6y9S8fXXxcOs'
        API_KEY="AIzaSyALxVs2qKta3_REg_epP3x6WFH1Q0lNfPM"
        self.client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
        )
        print("Client Load Done.")
        self.times=0

    def predict(self,text):
        analyze_request = {
        'comment': { 'text': text },
        'requestedAttributes': {'TOXICITY': {}}
        }
        try:
            response = self.client.comments().analyze(body=analyze_request).execute()
            res=json.dumps(response, indent=2)
            res=json.loads(res)
            value=float(res['attributeScores']["TOXICITY"]["summaryScore"]['value'])
        except:
            print(f"Error {self.times} times.")
            self.times+=1
            print(f"the incorrect query is: {text}")
            value=0.
        cls_res=0
        if value>0.5:
            cls_res=1
        return cls_res

## running entry
if __name__=="__main__":
    persAPI=PerspectiveAPI()
    res=persAPI.predict("Fucking everyone!")
    print(res)
    print("EVERYTHING DONE.")
