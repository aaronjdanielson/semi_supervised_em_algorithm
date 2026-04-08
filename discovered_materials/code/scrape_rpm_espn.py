import re
import urllib
from time import sleep

f = urllib.request.urlopen('https://www.espn.com/nba/statistics/rpm')
teams_source = f.read().decode('utf-8')


